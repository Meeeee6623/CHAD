import torch
import torch.nn as nn
import math
import warnings
from typing import Optional, Tuple, List
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
import types
from copy import deepcopy
import time

# --- Custom Config Namespace ---
class ConfigNamespace(types.SimpleNamespace):
    """SimpleNamespace with a .get() method and __contains__."""
    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        # Allow checking 'key in config_namespace' by seeing if the attribute exists
        return hasattr(self, key)

# --- RMSNorm (from scratch notebook) ---
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype=torch.float32):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype)) # Learnable weight

    def _norm(self, x):
        # Calculate RMS and normalize
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Cast to float for norm calculation stability, then cast back
        output = self._norm(x.float()).type_as(x)
        return output * self.weight # Apply learnable weight

# --- RoPE Functions (Adapted for device handling) ---
def precompute_rope_params(head_dim, context_length, theta_base=10000.0, freq_config=None, device=None, dtype=torch.float32):
    if freq_config: theta_base = freq_config.get("rope_base", theta_base)
    assert head_dim % 2 == 0
    theta = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float) / head_dim))
    positions = torch.arange(context_length, device=device, dtype=torch.float) # Use float positions
    freqs = torch.outer(positions, theta)
    # Don't compute freqs_cis, just return cos and sin directly
    cos = torch.cos(freqs).to(dtype) # [context_length, head_dim // 2]
    sin = torch.sin(freqs).to(dtype) # [context_length, head_dim // 2]
    return cos, sin

def compute_rope(x, cos, sin):
    # x shape: [B, H, T, Dh]
    batch_size, num_heads, seq_len, head_dim = x.shape
    device = x.device

    # Select cos/sin for the current sequence length
    cos_t = cos[:seq_len].unsqueeze(0).unsqueeze(0) # [1, 1, T, Dh/2]
    sin_t = sin[:seq_len].unsqueeze(0).unsqueeze(0) # [1, 1, T, Dh/2]

    x1 = x[..., : head_dim // 2] # [B, H, T, Dh/2]
    x2 = x[..., head_dim // 2 :] # [B, H, T, Dh/2]

    # Apply rotation using real arithmetic directly compatible with scratch implementation
    rotated_x2 = -x2
    rotated_x = torch.cat(
        (x1 * cos_t - x2 * sin_t, x2 * cos_t + x1 * sin_t), dim=-1
    )
    return rotated_x.to(x.dtype)


# --- Attention (Adapting scratch GQA) ---
class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_in = cfg.emb_dim
        d_out = cfg.emb_dim
        self.num_heads = cfg.n_heads
        self.num_kv_groups = cfg.n_kv_groups
        dtype = cfg.dtype

        assert d_out % self.num_heads == 0
        assert self.num_heads % self.num_kv_groups == 0

        self.head_dim = d_out // self.num_heads
        self.num_kv_heads = self.num_kv_groups # Renaming for clarity
        self.group_size = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(d_in, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(d_in, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_in, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, d_out, bias=False, dtype=dtype)

        # RoPE buffers - Initialized in _init_buffers
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        # Causal mask buffer
        self.register_buffer("mask", None, persistent=False)

        # Store cfg for _init_buffers
        self._config = cfg # Use a different name to avoid potential conflicts

        # Check if flash attention is enabled in config
        self.use_flash_attn = getattr(cfg, "use_flash_attn", False)

        # Check if flash attention is available
        self.flash_attn_available = False
        try:
            from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
            self.flash_attn_available = True
            self.flash_attn_func = flash_attn_func
            print(f"Flash Attention 2 is available and {'enabled' if self.use_flash_attn else 'disabled'} in config.")
        except ImportError:
            if self.use_flash_attn:
                print("Warning: Flash Attention 2 requested in config but not available in environment.")
                print("Please install flash-attn package: pip install flash-attn")
            self.use_flash_attn = False

    def _init_buffers(self, context_len, device):
        if self.cos_cached is None or self.cos_cached.device != device or self.cos_cached.shape[0] < context_len:
            print(f"Initializing RoPE/Mask buffers (len={context_len}) on device {device}")
            theta = self._config.rope_base
            self.cos_cached, self.sin_cached = precompute_rope_params(
                self.head_dim, context_len, theta_base=theta,
                freq_config=getattr(self._config, "rope_freq", None), # Use getattr for optional keys
                device=device, dtype=self._config.dtype
            )
            # Causal mask
            self.mask = torch.triu(torch.ones(context_len, context_len, dtype=torch.bool, device=device), diagonal=1)


    def forward(self, x, position_ids=None):
        b, num_tokens, d_in = x.shape
        device = x.device
        self._init_buffers(self._config.context_length, device)

        # 1. Project
        queries = self.q_proj(x)  # [B, T, H*Dh]
        keys = self.k_proj(x)      # [B, T, Hkv*Dh]
        values = self.v_proj(x)    # [B, T, Hkv*Dh]

        # 2. Reshape
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim) # [B, T, H, Dh]
        keys = keys.view(b, num_tokens, self.num_kv_heads, self.head_dim)    # [B, T, Hkv, Dh]
        values = values.view(b, num_tokens, self.num_kv_heads, self.head_dim) # [B, T, Hkv, Dh]

        # 3. Apply RoPE *before* transpose (RoPE needs [B, T, H, Dh] or similar)
        # Need position IDs - assume default if None
        if position_ids is None:
            position_ids = torch.arange(num_tokens, device=device).unsqueeze(0)
        else:
            position_ids = position_ids.to(device)

        # Select the cos/sin values corresponding to the position_ids
        max_pos = self.cos_cached.shape[0]
        if torch.any(position_ids >= max_pos):
            warnings.warn(f"Position IDs ({position_ids.max()}) exceed RoPE cache length ({max_pos}). Clamping.")
            position_ids = torch.clamp(position_ids, max=max_pos-1)

        cos = self.cos_cached[position_ids].unsqueeze(2) # [B, T, 1, Dh/2]
        sin = self.sin_cached[position_ids].unsqueeze(2) # [B, T, 1, Dh/2]

        # Apply rotary positional embeddings directly
        def apply_rotary_pos_emb(t, cos, sin):
            # t: [B, T, H, Dh]
            t_ = t.float() # Ensure float for precision
            t1 = t_[..., : self.head_dim // 2]
            t2 = t_[..., self.head_dim // 2 :]
            rotated = torch.cat(
                (t1 * cos - t2 * sin, t1 * sin + t2 * cos), dim=-1
            )
            return rotated.type_as(t)

        queries = apply_rotary_pos_emb(queries, cos, sin)
        keys = apply_rotary_pos_emb(keys, cos, sin)

        # 4. Transpose for attention calculation: [B, H, T, Dh]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 5. Repeat K/V heads *after* RoPE and transpose
        keys = keys.repeat_interleave(self.group_size, dim=1) # [B, H, T, Dh]
        values = values.repeat_interleave(self.group_size, dim=1) # [B, H, T, Dh]

        # 6. Compute attention
        if self.use_flash_attn and self.flash_attn_available:
            # Use Flash Attention 2
            # Flash attention expects inputs in [B, S, H, D] format
            # Need to transpose from [B, H, S, D] to [B, S, H, D]
            q = queries.transpose(1, 2)  # [B, T, H, Dh]
            k = keys.transpose(1, 2)     # [B, T, H, Dh]
            v = values.transpose(1, 2)   # [B, T, H, Dh]

            # Flash attention requires inputs to be contiguous
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

            # Compute attention with flash_attn_func
            # softmax_scale defaults to 1/sqrt(head_dim)
            context_vec = self.flash_attn_func(
                q, k, v,
                dropout_p=0.0,
                causal=True,
                softmax_scale=1.0 / math.sqrt(self.head_dim)  # Use math.sqrt for scalar value
            )  # [B, T, H, Dh]

            # Reshape to [B, T, H*Dh]
            context_vec = context_vec.reshape(b, num_tokens, self.num_heads * self.head_dim)
        else:
            # Fallback to regular attention
            # Transpose K for matmul
            keys_t = keys.transpose(-2, -1) # [B, H, Dh, T]

            # Compute attention scores
            attn_scores = queries @ keys_t

            # Apply causal mask
            mask_bool = self.mask[:num_tokens, :num_tokens] # Slice mask [T, T]
            # Expand mask to attention score shape [B, H, Tq, Tk]
            # No batch or head dim needed for broadcast if mask is [Tq, Tk]
            attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)

            # Apply softmax scaling
            attn_weights = torch.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1)  # Use math.sqrt for scalar value
            attn_weights = attn_weights.to(values.dtype)

            # Compute context vector
            context_vec = (attn_weights @ values).transpose(1, 2) # [B, T, H, Dh]
            context_vec = context_vec.reshape(b, num_tokens, self.num_heads * self.head_dim) # [B, T, E]

        # Apply output projection
        context_vec = self.o_proj(context_vec)

        return context_vec

# --- FeedForward (Adapting scratch FFN) ---
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Use Llama 3 SwiGLU names/structure from scratch code
        self.fc1 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, dtype=cfg.dtype, bias=False) # gate_proj
        self.fc2 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, dtype=cfg.dtype, bias=False) # up_proj
        self.fc3 = nn.Linear(cfg.hidden_dim, cfg.emb_dim, dtype=cfg.dtype, bias=False) # down_proj

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2 # SwiGLU activation
        return self.fc3(x)

# --- Transformer Block (Using scratch structure) ---
class TransformerBlock(nn.Module):
    def __init__(self, cfg): # No layer_idx needed here
        super().__init__()
        self.att = GroupedQueryAttention(cfg)
        self.ff = FeedForward(cfg)
        # Use RMSNorm from scratch code implementation
        self.norm1 = RMSNorm(cfg.emb_dim, eps=cfg.get("rms_norm_eps", 1e-5), dtype=cfg.dtype)
        self.norm2 = RMSNorm(cfg.emb_dim, eps=cfg.get("rms_norm_eps", 1e-5), dtype=cfg.dtype)

    def forward(self, x, position_ids=None): # Pass position_ids
        residual = x
        # Apply norm first, then attention (Pre-Norm)
        hidden_states_norm = self.norm1(x)
        attn_output = self.att(hidden_states_norm, position_ids=position_ids)
        h = residual + attn_output # Residual connection

        residual = h
        hidden_states_norm = self.norm2(h)
        ff_output = self.ff(hidden_states_norm)
        out = residual + ff_output # Residual connection

        return out

# --- Hopfield Layer (Refined version) ---
class HopfieldMemoryLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg # Store config SimpleNamespace directly
        self.emb_dim = cfg.emb_dim
        self.n_heads = cfg.hopfield_heads
        self.n_memory_slots = cfg.hopfield_memory_slots

        # Use explicit head_dim if provided, otherwise calculate from emb_dim
        self.head_dim = getattr(cfg, "hopfield_head_dim", self.emb_dim // self.n_heads)
        print(f"[DEBUG] Using Hopfield head_dim: {self.head_dim} ({'explicitly set' if hasattr(cfg, 'hopfield_head_dim') else 'calculated from emb_dim'})")

        # Pattern dim is now head_dim * n_heads (might be different from emb_dim if head_dim is explicit)
        self.pattern_dim = self.head_dim * self.n_heads

        # Add projection layer if pattern_dim != emb_dim
        self.needs_dim_projection = self.pattern_dim != self.emb_dim
        if self.needs_dim_projection:
            print(f"[DEBUG] Adding dimension projection layer: {self.emb_dim} -> {self.pattern_dim}")
            self.dim_proj = nn.Linear(self.emb_dim, self.pattern_dim, bias=False, dtype=self.dtype)
            # Also add inverse projection for converting back to emb_dim when needed
            print(f"[DEBUG] Adding inverse dimension projection layer: {self.pattern_dim} -> {self.emb_dim}")
            self.inv_dim_proj = nn.Linear(self.pattern_dim, self.emb_dim, bias=False, dtype=self.dtype)

        self.num_updates = cfg.hopfield_num_updates
        # Use getattr for optional config keys to provide defaults
        self.update_strategy = getattr(cfg, "hopfield_update_strategy", "none")
        self.memory_update_lr = getattr(cfg, "hopfield_memory_update_lr", 0.01)
        self.gate_input_pooling = getattr(cfg, "hopfield_gate_input_pooling", "mean")
        self.update_target_method = getattr(cfg, "hopfield_update_target_method", "avg_query")
        self.clamp_patterns = getattr(cfg, "hopfield_clamp_patterns", None) # Default None
        self.clamp_beta = getattr(cfg, "hopfield_clamp_beta", 10.0)
        self.combine_method = getattr(cfg, "hopfield_combine_method", "add")
        self.init_method = getattr(cfg, "hopfield_init_method", "random")
        self.dtype = getattr(cfg, "dtype", torch.float32) # Default float32

        # Check if flash attention is enabled
        self.use_flash_attn = getattr(cfg, "attn_implementation", "") == "flash_attention_2"
        self.flash_attn_available = False
        if self.use_flash_attn:
            # Try to import flash attention
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
                self.flash_attn_available = True
                print(f"[DEBUG] Flash Attention 2 is ENABLED for Hopfield memory calculations")
            except ImportError:
                self.use_flash_attn = False
                print(f"[DEBUG] Flash Attention 2 is requested but not available. Falling back to standard attention.")
        else:
            print(f"[DEBUG] Using standard attention for Hopfield memory calculations")

        self.storedpatterns = nn.Parameter(
            torch.zeros(self.n_heads, self.n_memory_slots, self.head_dim),
            requires_grad=cfg.get("hopfield_learn_patterns", False)
        )

        # Print debug information about parameter sizes
        storedpatterns_params = self.n_heads * self.n_memory_slots * self.head_dim
        print(f"[DEBUG] Hopfield storedpatterns: shape={self.storedpatterns.shape}, params={storedpatterns_params}")

        # Core projection matrices
        self.q_proj = nn.Linear(self.emb_dim, self.n_heads * self.head_dim, bias=cfg.get("hopfield_use_bias", False), dtype=self.dtype)
        self.k_proj = nn.Linear(self.head_dim, self.head_dim, bias=cfg.get("hopfield_use_bias", False), dtype=self.dtype)
        self.v_proj = nn.Linear(self.head_dim, self.head_dim, bias=cfg.get("hopfield_use_bias", False), dtype=self.dtype)

        # Print linear layer parameter counts
        q_proj_params = self.emb_dim * (self.n_heads * self.head_dim)
        k_proj_params = self.head_dim * self.head_dim
        v_proj_params = self.head_dim * self.head_dim
        print(f"[DEBUG] Hopfield q_proj: params={q_proj_params}")
        print(f"[DEBUG] Hopfield k_proj: params={k_proj_params}")
        print(f"[DEBUG] Hopfield v_proj: params={v_proj_params}")

        self._initialized = False

        # Correct beta shape for broadcasting: [1, H_hop, 1, 1]
        self.beta = nn.Parameter(torch.ones(1, self.n_heads, 1, 1, dtype=self.dtype))

        # Match Norm type with base model
        self.norm_query = RMSNorm(self.emb_dim, eps=cfg.get("rms_norm_eps", 1e-5), dtype=self.dtype)
        self.norm_retrieved = RMSNorm(self.pattern_dim, eps=cfg.get("rms_norm_eps", 1e-5), dtype=self.dtype)

        # Add embedding projection layer for dimension mismatches during sampling
        self.embedding_proj = nn.Linear(self.emb_dim, self.n_heads * self.head_dim, bias=False, dtype=self.dtype)

        if self.update_strategy == "gated":
            gate_input_dim = self.emb_dim * 2
            gate_output_dim = self.n_heads * self.n_memory_slots * self.head_dim
            self.gate_linear = nn.Linear(gate_input_dim, gate_output_dim, bias=False, dtype=self.dtype)
            print(f"[DEBUG] Hopfield gate_linear: params={gate_input_dim * gate_output_dim}")
            print(f"Initialized Gated Update mechanism.")

        # Always create combine_proj if method is concat
        if self.combine_method == "concat":
            self.combine_proj = nn.Linear(self.emb_dim + self.pattern_dim, self.emb_dim, bias=False, dtype=self.dtype)
            print(f"[DEBUG] Hopfield combine_proj: params={(self.emb_dim + self.pattern_dim) * self.emb_dim}")
            print("Initialized Concat combine method.")
        # Always create combine_gate and query_combine_proj if method is gated_add
        elif self.combine_method == "gated_add":
            # Determine the input dimension for combine_gate (concatenated query and retrieved memory)
            # Query might need projection if pattern_dim != emb_dim
            gate_input_dim = self.pattern_dim * 2
            if self.pattern_dim != self.emb_dim:
                # Need query_combine_proj layer
                self.query_combine_proj = nn.Linear(self.emb_dim, self.pattern_dim, bias=False, dtype=self.dtype)
                print("[DEBUG] Initialized query_combine_proj for gated_add")

            # Create the combine_gate layer
            self.combine_gate = nn.Linear(gate_input_dim, self.pattern_dim, bias=False, dtype=self.dtype)
            print("[DEBUG] Initialized combine_gate for gated_add")
            print("Initialized Gated Add combine method layers.")

        # Calculate and print total parameter count
        total_params = storedpatterns_params + q_proj_params + k_proj_params + v_proj_params + self.n_heads
        if self.needs_dim_projection:
            total_params += self.emb_dim * self.pattern_dim  # dim_proj
            total_params += self.pattern_dim * self.emb_dim  # inv_dim_proj
        if self.update_strategy == "gated":
            total_params += gate_input_dim * gate_output_dim
        if self.combine_method == "concat":
            total_params += (self.emb_dim + self.pattern_dim) * self.emb_dim
        if self.combine_method == "gated_add":
            # Add combine_gate param count
            gate_input_dim = self.pattern_dim * 2
            total_params += gate_input_dim * self.pattern_dim
            # Add query_combine_proj param count if needed
            if self.pattern_dim != self.emb_dim:
                total_params += self.emb_dim * self.pattern_dim

        print(f"[DEBUG] Hopfield total trainable parameters: {total_params}")

        self._clear_last_state()

    def initialize_memory(self, embedding_matrix: Optional[torch.Tensor] = None):
        """Initialize Hopfield memory with either random values or sampled from embedding matrix."""
        print(f"[DEBUG] Starting Hopfield memory initialization with method: {self.init_method}")
        if self._initialized: return # Skip if already initialized

        # Initialize the patterns and temperature parameter based on initialization method
        if self.init_method == "uniform":
            print(f"[DEBUG] Using uniform initialization")
            nn.init.uniform_(self.storedpatterns, -0.02, 0.02)
            print(f"[DEBUG] Uniform initialization complete")
        elif self.init_method == "kaiming_uniform":
            print(f"[DEBUG] Using kaiming_uniform initialization")
            nn.init.kaiming_uniform_(self.storedpatterns, a=math.sqrt(5))
            print(f"[DEBUG] Kaiming initialization complete")
        elif self.init_method == "embedding_sampling":
            print(f"[DEBUG] Using embedding_sampling initialization")
            if embedding_matrix is not None:
                print(f"[DEBUG] Embedding matrix provided, shape={embedding_matrix.shape}")
                # Sample from embedding entries, reshape to [n_heads, n_slots, head_dim]
                try:
                    # Sample random indices from embedding matrix for each memory slot
                    vocab_size = embedding_matrix.shape[0]
                    total_slots = self.n_heads * self.n_memory_slots

                    # Create indices for sampling
                    indices = torch.randint(0, vocab_size, (total_slots,), device=embedding_matrix.device)

                    # Sample embeddings
                    samples = embedding_matrix[indices]
                    print(f"[DEBUG] Sampled embeddings shape: {samples.shape}")

                    # Check dimensions
                    emb_dim = embedding_matrix.shape[1]
                    print(f"[DEBUG] Embedding dim: {emb_dim}, Pattern dim: {self.pattern_dim}, Head dim: {self.head_dim}")

                    # Project the embeddings if dimensions don't match
                    if emb_dim != self.pattern_dim:
                        print(f"[DEBUG] Projecting embeddings to match pattern dimension")
                        if self.needs_dim_projection:
                            samples = self.dim_proj(samples)
                        else:
                            # If we don't have a projection layer but sizes differ, use embedding_proj
                            samples = self.embedding_proj(samples)

                    # Reshape to [n_heads, n_slots, head_dim]
                    try:
                        # Ensure the samples tensor matches the exact size needed for reshaping
                        total_size = self.n_heads * self.n_memory_slots * self.head_dim
                        if samples.numel() != total_size:
                            print(f"[DEBUG] Sample size mismatch. Have: {samples.numel()}, Need: {total_size}")
                            # Pad or truncate as needed to match the required size
                            samples_flat = samples.reshape(-1)
                            if samples_flat.size(0) < total_size:
                                # Pad if too small
                                print(f"[DEBUG] Padding samples from {samples_flat.size(0)} to {total_size}")
                                pad_size = total_size - samples_flat.size(0)
                                padding = torch.randn(pad_size, device=samples_flat.device, dtype=samples_flat.dtype) * 0.02
                                samples_flat = torch.cat([samples_flat, padding])
                            else:
                                # Truncate if too large
                                print(f"[DEBUG] Truncating samples from {samples_flat.size(0)} to {total_size}")
                                samples_flat = samples_flat[:total_size]

                            # Now reshape to target dimensions
                            samples = samples_flat.reshape(self.n_heads, self.n_memory_slots, self.head_dim)
                        else:
                            # Reshape directly if size matches
                            samples = samples.reshape(self.n_heads, self.n_memory_slots, self.head_dim)

                        print(f"[DEBUG] Reshaped to {samples.shape}")
                        self.storedpatterns.data.copy_(samples)
                        print(f"[DEBUG] Successfully copied embeddings to stored patterns")
                    except RuntimeError as e:
                        # If reshaping still fails, try an alternative approach
                        print(f"[DEBUG] Reshape failed: {e}. Trying alternative approach")
                        # Create a completely new tensor with the right shape
                        print(f"[DEBUG] Creating new tensor with shape ({self.n_heads}, {self.n_memory_slots}, {self.head_dim})")
                        new_patterns = torch.randn(
                            self.n_heads, self.n_memory_slots, self.head_dim,
                            device=embedding_matrix.device,
                            dtype=self.dtype
                        ) * 0.02
                        self.storedpatterns.data.copy_(new_patterns)
                        print(f"[DEBUG] Successfully created new random patterns with proper shape")
                except Exception as e:
                    warnings.warn(f"Error in embedding sampling: {e}. Falling back to random.")
                    print(f"[DEBUG] Error: {str(e)}")
                    self.storedpatterns.data.normal_(mean=0.0, std=0.02)
                    print(f"[DEBUG] Initialized Hopfield memory with random values")
            else:
                if self.init_method == "embedding_sampling":
                    warnings.warn("Embedding matrix not provided for 'embedding_sampling' init. Using random.")
                print(f"[DEBUG] Using random initialization")
                self.storedpatterns.data.normal_(mean=0.0, std=0.02)
                print(f"[DEBUG] Random initialization complete")
        self._initialized = True
        print(f"[DEBUG] Hopfield memory initialization complete")


    def forward(self, query_input): # Removed mask pass-through for simplicity
        if not self._initialized: raise RuntimeError("Hopfield memory not initialized.")

        batch_size, seq_len, _ = query_input.shape
        query_input_dtype = query_input.dtype
        device = query_input.device

        # Memory optimization: only convert dtype if needed
        if query_input.dtype != self.dtype:
            query_input = query_input.to(self.dtype)

        # Apply dimension projection if needed
        if self.needs_dim_projection:
            query_input = self.dim_proj(query_input)

        # Reuse memory for query normalization
        query_norm = self.norm_query(query_input)

        # Project query (memory optimization: avoid creating intermediate tensors)
        query = self.q_proj(query_norm).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Move stored patterns to device once (memory optimization: cache on appropriate device)
        if not hasattr(self, '_cached_patterns_device') or self._cached_patterns_device != device:
            self.storedpatterns_device = self.storedpatterns.to(device=device, dtype=self.dtype)
            self._cached_patterns_device = device

        # Always ensure storedpatterns_device is defined
        storedpatterns_device = self.storedpatterns_device

        # Process keys and values (memory optimization: combine operations)
        keys_values_flat = storedpatterns_device.view(-1, self.head_dim)
        keys_flat = self.k_proj(keys_values_flat)
        values_flat = self.v_proj(keys_values_flat)
        keys = keys_flat.view(1, self.n_heads, self.n_memory_slots, self.head_dim)
        values = values_flat.view(1, self.n_heads, self.n_memory_slots, self.head_dim)

        # Precompute beta tensor only once (memory optimization: avoid repeated to() calls)
        beta_device = self.beta.to(device)

        if self.clamp_beta is not None:
            with torch.no_grad():
                self.beta.data.clamp_(min=1e-2, max=self.clamp_beta)

        # Memory optimization: store with detach() not clone() during training
        if self.training and self.update_strategy != "none":
            self._last_query_input = query_input.detach()
            self._last_query_proj = query.detach()

        # Use Flash Attention 2 if available and enabled
        if self.use_flash_attn and self.flash_attn_available:
            # Prepare for flash attention
            q_flash = query.transpose(1, 2)
            k_flash = keys.expand(batch_size, -1, -1, -1).transpose(1, 2)
            v_flash = values.expand(batch_size, -1, -1, -1).transpose(1, 2)

            # Scale query by beta - ensure proper broadcasting
            beta_value = beta_device.squeeze(-1).squeeze(-1)  # [1, H]
            # Reshape beta to match q_flash's head dimension
            beta_value = beta_value.unsqueeze(0).unsqueeze(-1)  # [1, 1, H, 1]
            q_flash = q_flash * beta_value

            # Precompute scaling factor
            scale = float(1.0 / math.sqrt(self.head_dim))

            # Try flash attention with error handling
            try:
                retrieved_memory = self.flash_attn_func(
                    q_flash, k_flash, v_flash,
                    dropout_p=0.0,
                    causal=False,
                    softmax_scale=scale
                )
                retrieved_memory = retrieved_memory.transpose(1, 2)
            except Exception as e:
                print(f"[WARNING] Flash attention failed: {e}. Using standard attention.")
                # Fall back to standard attention
                attn_scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
                attn_scores = attn_scores * beta_device
                attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(self.dtype)
                retrieved_memory = torch.matmul(attn_weights, values)

                if self.training and self.update_strategy != "none":
                    self._last_attn_weights = attn_weights.detach()
        else:
            # Standard attention (memory optimization: fuse operations where possible)
            attn_scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_scores = attn_scores * beta_device
            attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(self.dtype)
            retrieved_memory = torch.matmul(attn_weights, values)

            if self.training and self.update_strategy != "none":
                self._last_attn_weights = attn_weights.detach()

        # Memory optimization: fuse reshape operations
        retrieved_memory_t = retrieved_memory.transpose(1, 2).reshape(batch_size, seq_len, self.pattern_dim)
        final_retrieved_norm = self.norm_retrieved(retrieved_memory_t)

        if self.training and self.update_strategy != "none":
            self._last_retrieved_memory_norm = final_retrieved_norm.detach()

        # Combine retrieved memory with original query based on configured method
        if self.combine_method == "add":
            if self.needs_dim_projection:
                final_retrieved_norm = self.inv_dim_proj(final_retrieved_norm)
            memory_output = query_input + final_retrieved_norm
        elif self.combine_method == "gated_add":
            # Gating mechanism should have been created in __init__
            # Ensure the layers exist before using them
            if not hasattr(self, 'combine_gate') or (self.pattern_dim != self.emb_dim and not hasattr(self, 'query_combine_proj')):
                raise RuntimeError("gated_add combine method selected, but required layers (combine_gate/query_combine_proj) not initialized.")

            # Use existing gating mechanism
            if self.pattern_dim != self.emb_dim:
                projected_query = self.query_combine_proj(query_input)
            else:
                projected_query = query_input
            # Concatenate for gating computation
            gate_concat = torch.cat([projected_query, final_retrieved_norm], dim=-1)
            # Create gate value between 0-1 for how much to use memory vs. query
            gate_value = torch.sigmoid(self.combine_gate(gate_concat))
            # Apply gate and project back if needed
            gated_memory = gate_value * final_retrieved_norm
            if self.needs_dim_projection:
                gated_memory = self.inv_dim_proj(gated_memory)
            memory_output = query_input + gated_memory
        elif self.combine_method == "concat":
            # Concatenate and project back to emb_dim
            concat_output = torch.cat([query_input, final_retrieved_norm], dim=-1)
            memory_output = self.combine_proj(concat_output)
        else:
            # Default/unknown option - Just use retrieved memory with proper dim projection
            if self.needs_dim_projection:
                final_retrieved_norm = self.inv_dim_proj(final_retrieved_norm)
            memory_output = final_retrieved_norm

        # Return correctly typed output (memory optimization: only convert if needed)
        if memory_output.dtype != query_input_dtype:
            memory_output = memory_output.to(query_input_dtype)
        return memory_output

    def update_memory(self):
        if self.update_strategy in ["none", None] or self._last_query_input is None:
            self._clear_last_state(); return

        with torch.no_grad():
            device = self.storedpatterns.device
            dtype = self.storedpatterns.dtype
            update_delta = None

            # Calculate Update Target
            update_target = None
            if self.update_target_method == "avg_query":
                if self._last_query_input is not None:
                    # Optimize: Move to device once with correct dtype
                    last_query_device = self._last_query_input.to(device, dtype)
                    avg_query = torch.mean(last_query_device, dim=(0, 1))
                    try:
                        target_per_head = avg_query.view(self.n_heads, self.head_dim)
                        update_target = target_per_head.unsqueeze(1).expand_as(self.storedpatterns)
                    except RuntimeError as e: print(f"Error reshaping avg_query: {e}")
                else: print("Warning: Missing query input for update target.")
            # Add other methods ('avg_retrieved', 'attention_weighted_query') here if implementing
            elif self.update_target_method == "avg_retrieved":
                if self._last_retrieved_memory_norm is not None:
                    # Optimize: Move to device once with correct dtype
                    last_retrieved_device = self._last_retrieved_memory_norm.to(device, dtype)
                    avg_retrieved = torch.mean(last_retrieved_device, dim=(0, 1))
                    try:
                        target_per_head = avg_retrieved.view(self.n_heads, self.head_dim)
                        update_target = target_per_head.unsqueeze(1).expand_as(self.storedpatterns)
                    except RuntimeError as e: print(f"Error reshaping avg_retrieved: {e}")
                else: print("Warning: Missing retrieved memory for 'avg_retrieved' target.")
            # elif self.update_target_method == "attention_weighted_query": # Implementation omitted for brevity


            if update_target is not None:
                potential_update_delta = (update_target - self.storedpatterns.data).to(dtype)
            else:
                potential_update_delta = None

            # Apply Gating
            if self.update_strategy == "gated" and potential_update_delta is not None:
                if self._last_retrieved_memory_norm is None:
                    print("Warning: Missing inputs for gated update calculation.")
                    update_delta = None
                else:
                    query_pooled, retrieved_pooled = None, None
                    # Optimize: Move to device once for each tensor
                    last_query_device = self._last_query_input.to(device)
                    last_retrieved_device = self._last_retrieved_memory_norm.to(device)

                    if self.gate_input_pooling == "mean":
                        query_pooled = torch.mean(last_query_device, dim=(0, 1))
                        retrieved_pooled = torch.mean(last_retrieved_device, dim=(0, 1))
                    elif self.gate_input_pooling == "max":
                        query_pooled = torch.max(last_query_device, dim=1)[0].mean(dim=0)
                        retrieved_pooled = torch.max(last_retrieved_device, dim=1)[0].mean(dim=0)

                    if query_pooled is not None and retrieved_pooled is not None:
                        gate_input = torch.cat((query_pooled, retrieved_pooled), dim=-1).to(self.gate_linear.weight.dtype)
                        gate_values_flat = torch.sigmoid(self.gate_linear(gate_input))
                        gate = gate_values_flat.view_as(self.storedpatterns).to(dtype)
                        update_delta = gate * potential_update_delta
                    else:
                        update_delta = None
            elif self.update_strategy == "persistent":
                update_delta = potential_update_delta

            # Apply Final Update
            if update_delta is not None:
                self.storedpatterns.data.add_(update_delta, alpha=self.memory_update_lr)
                if self.clamp_patterns is not None:
                    self.storedpatterns.data.clamp_(min=-self.clamp_patterns, max=self.clamp_patterns) # In-place clamp

        self._clear_last_state()


    def _clear_last_state(self):
        self._last_attn_weights = None
        self._last_query_input = None
        self._last_query_proj = None
        self._last_retrieved_memory_norm = None

    def reset_memory(self):
        """Clears the internal state used for memory updates to prevent state leakage between examples."""
        # Clear all internal state variables
        self._clear_last_state()

        # Important: Detach all computational history from storedpatterns to break dependency chains
        # This is critical for preventing gradient accumulation chains across batch examples
        if hasattr(self, 'storedpatterns'):
            # Clone and detach to ensure no computational graph connections remain
            # While preserving the actual learned values
            with torch.no_grad():
                detached_patterns = self.storedpatterns.clone().detach()
                self.storedpatterns.data.copy_(detached_patterns)

        # Detach any other gradient-related state
        if hasattr(self, 'beta'):
            with torch.no_grad():
                detached_beta = self.beta.clone().detach()
                self.beta.data.copy_(detached_beta)

        print("Hopfield memory state fully cleared and detached from computational graph.")


# --- HAT Model (Using scratch structure) ---
class HopfieldLlama3Model(nn.Module, GenerationMixin):
    main_input_name = "input_ids"

    def __init__(self, cfg):
        super().__init__()
        print(f"[DEBUG] Starting initialization of HopfieldLlama3Model with placement: {cfg.get('hopfield_layer_placement', 'pre_post')}")
        # Ensure cfg is a ConfigNamespace
        if isinstance(cfg, dict):
            cfg = ConfigNamespace(**cfg)
        self.cfg = cfg # Keep original ConfigNamespace if needed elsewhere
        self.config = cfg # Use the ConfigNamespace directly
        self.config.is_encoder_decoder = False # Explicitly set for GenerationMixin compatibility

        # --- REVISED APPROACH ---
        # Set use_cache on the config object (standard)
        self.config.use_cache = False
        # Set _supports_cache_class directly on the model instance
        self._supports_cache_class = False # <<< Set directly on self

        # Create embedding and token types
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.emb_dim, dtype=self.config.dtype)

        # Get the placement for Hopfield memory layer
        self.hopfield_layer_placement = self.config.get('hopfield_layer_placement', 'pre_post')

        # Determine middle split point if using pre_middle configuration
        self.pre_middle_split = self.config.get('pre_middle_split', self.config.n_layers // 2)
        print(f"[DEBUG] Using Hopfield placement: {self.hopfield_layer_placement}, pre_middle_split: {self.pre_middle_split}")

        # Create pre Hopfield memory layer if needed
        if self.hopfield_layer_placement in ['pre_post', 'pre_middle', 'pre_only']:
            print(f"[DEBUG] Creating pre_hopfield_memory layer")
            self.pre_hopfield_memory = HopfieldMemoryLayer(self.config)
        else:
            self.pre_hopfield_memory = None

        # Create transformer blocks
        self.trf_blocks = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.n_layers)])

        # Create middle Hopfield memory layer if needed
        if self.hopfield_layer_placement == 'pre_middle':
            print(f"[DEBUG] Creating middle_hopfield_memory layer after transformer block {self.pre_middle_split-1}")

            # Create middle Hopfield config by checking for middle-specific params or falling back to regular params
            middle_config = deepcopy(self.config)

            # Check for middle-specific configurations
            if hasattr(self.config, 'hopfield_middle_heads'):
                middle_config.hopfield_heads = self.config.hopfield_middle_heads
                print(f"[DEBUG] Using middle-specific heads: {middle_config.hopfield_heads}")

            if hasattr(self.config, 'hopfield_middle_memory_slots'):
                middle_config.hopfield_memory_slots = self.config.hopfield_middle_memory_slots
                print(f"[DEBUG] Using middle-specific memory slots: {middle_config.hopfield_memory_slots}")

            if hasattr(self.config, 'hopfield_middle_head_dim'):
                middle_config.hopfield_head_dim = self.config.hopfield_middle_head_dim
                print(f"[DEBUG] Using middle-specific head dim: {middle_config.hopfield_head_dim}")

            self.middle_hopfield_memory = HopfieldMemoryLayer(middle_config)
        else:
            self.middle_hopfield_memory = None

        # Create post Hopfield memory layer if needed
        if self.hopfield_layer_placement in ['pre_post', 'post_only']:
            print(f"[DEBUG] Creating post_hopfield_memory layer")
            self.post_hopfield_memory = HopfieldMemoryLayer(self.config)
        else:
            self.post_hopfield_memory = None

        # Create final normalization layer and output head
        self.output_norm = RMSNorm(self.config.emb_dim, eps=self.config.get("rms_norm_eps", 1e-5), dtype=self.config.dtype)
        self.out_head = nn.Linear(self.config.emb_dim, self.config.vocab_size, bias=False, dtype=self.config.dtype)

        # Tie weights
        self.out_head.weight = self.tok_emb.weight

        # Enable gradient checkpointing if configured
        if self.config.get("gradient_checkpointing", False):
            self.gradient_checkpointing_enable()

        print("[DEBUG] HopfieldLlama3Model initialization complete")

    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings and tied output head."""
        old_embeddings = self.tok_emb
        self.tok_emb = nn.Embedding(new_num_tokens, self.config.emb_dim, dtype=self.config.dtype)

        # Copy old weights
        if old_embeddings is not None:
            self.tok_emb.weight.data[:old_embeddings.weight.size(0), :] = old_embeddings.weight.data

        # Update config
        self.config.vocab_size = new_num_tokens
        self.cfg['vocab_size'] = new_num_tokens

        # Retie weights
        self.out_head = nn.Linear(self.config.emb_dim, new_num_tokens, bias=False, dtype=self.config.dtype)
        self.out_head.weight = self.tok_emb.weight

        return self

    @property
    def device(self) -> torch.device:
        # Return the device of the first parameter found
        try:
            return next(self.parameters()).device
        except StopIteration:
            # Handle case where model has no parameters or is on meta device
            # Try checking buffers if no parameters
            try:
                return next(self.buffers()).device
            except StopIteration:
                # Default to CPU if no parameters or buffers found
                return torch.device("cpu")

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.Tensor]] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs):
        # input_ids: [B, T]
        # attention_mask: [B, T]
        # inputs_embeds: [B, T, E]

        # Determine whether to output attentions/hidden states based on args or config
        # Note: We should check the passed `use_cache` argument if we were implementing cache
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", False) # Respect passed argument if available
        output_attentions = output_attentions if output_attentions is not None else getattr(self.config, "output_attentions", False)
        output_hidden_states = output_hidden_states if output_hidden_states is not None else getattr(self.config, "output_hidden_states", False)
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", True) # HF usually defaults use_return_dict to True

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Generate or validate position IDs
        if position_ids is None and input_ids is not None:
            position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)

        # Get embeddings based on what's provided
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.tok_emb(input_ids)

        # Apply pre Hopfield memory layer if configured
        if self.pre_hopfield_memory is not None:
            h = self.pre_hopfield_memory(h)

        # Store hidden states if needed
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Process the first part of transformer blocks
        first_block_end = self.pre_middle_split if self.hopfield_layer_placement == 'pre_middle' else len(self.trf_blocks)

        for i in range(first_block_end):
            if output_hidden_states:
                all_hidden_states += (h,)

            # Apply gradient checkpointing if enabled
            if self.config.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    self.trf_blocks[i], h, position_ids, use_reentrant=False
                )
            else:
                h = self.trf_blocks[i](h, position_ids=position_ids)

            if output_attentions:
                all_self_attns += (self.trf_blocks[i].att.last_attn_weights,)

        # Apply middle Hopfield memory layer if configured
        if self.middle_hopfield_memory is not None:
            h = self.middle_hopfield_memory(h)

        # Process the second part of transformer blocks (if using pre_middle)
        if self.hopfield_layer_placement == 'pre_middle':
            for i in range(self.pre_middle_split, len(self.trf_blocks)):
                if output_hidden_states:
                    all_hidden_states += (h,)

                # Apply gradient checkpointing if enabled
                if self.config.gradient_checkpointing and self.training:
                    h = torch.utils.checkpoint.checkpoint(
                        self.trf_blocks[i], h, position_ids, use_reentrant=False
                    )
                else:
                    h = self.trf_blocks[i](h, position_ids=position_ids)

                if output_attentions:
                    all_self_attns += (self.trf_blocks[i].att.last_attn_weights,)

        # Apply post Hopfield memory layer if configured
        if self.post_hopfield_memory is not None:
            h = self.post_hopfield_memory(h)

        # Final layer norm and output projection
        h = self.output_norm(h)
        logits = self.out_head(h)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

            # Calculate cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape(-1, self.config.vocab_size),
                shift_labels.reshape(-1)
            )

        # Return based on return_dict flag
        if not return_dict:
            outputs = (logits,) + (None,) + (all_hidden_states,) + (all_self_attns,)
            return (loss,) + outputs if loss is not None else outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # Handles position_ids dynamically based on kwargs
        model_inputs = {"input_ids": input_ids}

        # # Get generation_start_position from kwargs if present
        # start_pos = kwargs.get("generation_start_position", 0) # Default to 0 if not provided

        seq_length = input_ids.shape[1]

        # Calculate position_ids starting from start_pos
        # position_ids = torch.arange(start_pos, start_pos + seq_length, device=input_ids.device).unsqueeze(0)

        # *** MODIFICATION: Always compute position_ids from 0 for use_cache=False ***
        # This ensures alignment with RoPE cache and attention mask when not using KV cache.
        position_ids = torch.arange(0, seq_length, device=input_ids.device).unsqueeze(0)
        # **************************************************************************

        model_inputs['position_ids'] = position_ids

        # Add attention_mask if available
        if attention_mask is not None:
            model_inputs['attention_mask'] = attention_mask

        # Add past_key_values if available (though not used in current implementation)
        if past_key_values is not None:
            model_inputs['past_key_values'] = past_key_values

        # Remove generation_start_position from kwargs if it exists,
        # as it's not expected by the model's forward pass directly
        kwargs.pop("generation_start_position", None)

        # Add remaining kwargs
        model_inputs.update(kwargs)

        return model_inputs

    def initialize_hopfield_memory(self):
        """Initialize Hopfield memory layers if enabled."""
        print("[DEBUG] Starting Hopfield memory initialization...")

        embedding_weight = self.tok_emb.weight

        # Initialize pre Hopfield memory if exists
        if self.pre_hopfield_memory is not None and hasattr(self.pre_hopfield_memory, "initialize_memory"):
            print("[DEBUG] Initializing pre Hopfield memory...")
            self.pre_hopfield_memory.initialize_memory(embedding_weight)

        # Initialize middle Hopfield memory if exists
        if self.middle_hopfield_memory is not None and hasattr(self.middle_hopfield_memory, "initialize_memory"):
            print("[DEBUG] Initializing middle Hopfield memory...")
            self.middle_hopfield_memory.initialize_memory(embedding_weight)

        # Initialize post Hopfield memory if exists
        if self.post_hopfield_memory is not None and hasattr(self.post_hopfield_memory, "initialize_memory"):
            print("[DEBUG] Initializing post Hopfield memory...")
            self.post_hopfield_memory.initialize_memory(embedding_weight)

        print("[DEBUG] Hopfield memory initialization complete")

    def gradient_checkpointing_enable(self):
        self.config.gradient_checkpointing = True
        print("Gradient checkpointing enabled for Transformer blocks.")

    def gradient_checkpointing_disable(self):
        self.config.gradient_checkpointing = False
        print("Gradient checkpointing disabled.")

    def can_generate(self) -> bool:
        return True

    def _validate_model_class(self):
        return

    def reset_hopfield_memory(self):
        """Reset all Hopfield memory components in the model to prevent information leakage between examples."""
        print("[DEBUG] Resetting all Hopfield memory components in HopfieldLlama3Model")

        reset_count = 0

        # Reset pre Hopfield memory if exists
        if hasattr(self, 'pre_hopfield_memory') and hasattr(self.pre_hopfield_memory, 'reset_memory'):
            self.pre_hopfield_memory.reset_memory()
            reset_count += 1
            print("[DEBUG] Reset pre_hopfield_memory")

        # Reset middle Hopfield memory if exists
        if hasattr(self, 'middle_hopfield_memory') and hasattr(self.middle_hopfield_memory, 'reset_memory'):
            self.middle_hopfield_memory.reset_memory()
            reset_count += 1
            print("[DEBUG] Reset middle_hopfield_memory")

        # Reset post Hopfield memory if exists
        if hasattr(self, 'post_hopfield_memory') and hasattr(self.post_hopfield_memory, 'reset_memory'):
            self.post_hopfield_memory.reset_memory()
            reset_count += 1
            print("[DEBUG] Reset post_hopfield_memory")

        print(f"[DEBUG] Reset {reset_count} Hopfield memory components in HopfieldLlama3Model")

        return self  # Return self for method chaining

# --- Function to create model and load weights ---
def create_model_and_load_weights(config: dict, use_lora: bool, lora_config_dict: Optional[dict] = None):
    import yaml
    from safetensors.torch import load_file as load_safetensors
    from huggingface_hub import hf_hub_download
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    import time

    print("[DEBUG] Starting model creation and weight loading...")
    start_time = time.time()

    # --- Ensure critical numeric params from config dict are floats ---
    try: config['rms_norm_eps'] = float(config.get('rms_norm_eps', 1e-5))
    except (ValueError, TypeError): raise TypeError(f"Config 'rms_norm_eps' must be a number.")
    try: config['rope_base'] = float(config.get('rope_base', 500000.0))
    except (ValueError, TypeError): raise TypeError(f"Config 'rope_base' must be a number.")

    # Determine torch dtype
    dtype_str = config.get("model_dtype", "float32")
    if dtype_str == "bfloat16" and torch.cuda.is_bf16_supported(): dtype = torch.bfloat16
    elif dtype_str == "float16": dtype = torch.float16
    else: dtype = torch.float32
    config["dtype"] = dtype
    print(f"[DEBUG] Using dtype: {dtype}")

    # Convert config dict to ConfigNamespace
    config = ConfigNamespace(**config)

    # Determine model class based on configuration
    placement = config.get('hopfield_layer_placement', 'pre_post')
    # *** ADD model_type EXPLICITLY TO CONFIG ***
    if placement == 'deep':
        print(f"[DEBUG] Initializing DeepHopfieldLlama3Model (deep Hopfield)")
        # Set model_type for DeepHopfieldLlama3Model
        config.model_type = 'deep_hopfield_llama'
        model = DeepHopfieldLlama3Model(config)
    else:
        print(f"[DEBUG] Initializing HopfieldLlama3Model with placement: {placement}")
        # Set model_type for standard HopfieldLlama3Model
        config.model_type = 'hopfield_llama' # Use a distinct name
        model = HopfieldLlama3Model(config)
    print(f"[DEBUG] Model class initialized (type: {config.model_type}) in {time.time() - start_time:.2f}s")

    # Load pretrained weights from HF Hub into OUR structure
    print(f"[DEBUG] Starting to load pretrained weights for {config.model_name}...")
    try:
        # Handle sharded weights
        num_shards = config.get("num_weight_shards", 2)
        print(f"[DEBUG] Attempting to load {num_shards} weight shards for {config.model_name}.")
        model_files_info = [(f"model-{i+1:05d}-of-{num_shards:05d}.safetensors" if num_shards > 1 else "model.safetensors") for i in range(num_shards)]

        state_dict = {}
        for i, filename in enumerate(model_files_info):
            shard_start = time.time()
            print(f"[DEBUG] Loading shard {i+1}/{num_shards}: {filename}")
            weights_file = hf_hub_download(
                repo_id=config.model_name,
                filename=filename,
                cache_dir=config.get("data_cache_dir", "./model_cache"),
            )
            print(f"[DEBUG] Downloaded {filename} to {weights_file}, loading to CPU...")
            shard_state_dict = load_safetensors(weights_file, device="cpu")
            print(f"[DEBUG] Shard {i+1} loaded with {len(shard_state_dict)} keys in {time.time() - shard_start:.2f}s")
            state_dict.update(shard_state_dict)
            del shard_state_dict # Free memory
            print(f"[DEBUG] Shard {i+1} merged into state_dict, now has {len(state_dict)} keys")

    except Exception as e:
        print(f"[DEBUG] Error loading weights: {str(e)}")
        raise RuntimeError(f"Failed to load weights: {e}. Check model name and HF Hub access.")

    print(f"[DEBUG] All shards loaded in {time.time() - start_time:.2f}s, starting key adaptation...")
    # Adapt HF state_dict keys to our scratch model names
    adapted_state_dict = {}
    keys_adapted = 0

    # Determine model type to adapt keys correctly
    is_deep_model = placement == 'deep'

    for key, value in state_dict.items():
        new_key = key
        if key == "model.embed_tokens.weight": new_key = "tok_emb.weight"
        elif key.startswith("model.layers."):
            parts = key.split('.')
            layer_idx = parts[2]; layer_type = parts[3]; param_name = parts[-1]; sub_module = parts[4]
            if layer_type == "self_attn":
                hf_to_scratch_attn = {"q_proj": "att.q_proj", "k_proj": "att.k_proj", "v_proj": "att.v_proj", "o_proj": "att.o_proj"}
                if sub_module in hf_to_scratch_attn: new_key = f"trf_blocks.{layer_idx}.{hf_to_scratch_attn[sub_module]}.{param_name}"
            elif layer_type == "mlp":
                hf_to_scratch_ffn = {"gate_proj": "ff.fc1", "up_proj": "ff.fc2", "down_proj": "ff.fc3"}
                if sub_module in hf_to_scratch_ffn: new_key = f"trf_blocks.{layer_idx}.{hf_to_scratch_ffn[sub_module]}.{param_name}"
            elif layer_type == "input_layernorm": new_key = f"trf_blocks.{layer_idx}.norm1.{param_name}"
            elif layer_type == "post_attention_layernorm": new_key = f"trf_blocks.{layer_idx}.norm2.{param_name}"
        elif key == "model.norm.weight":
            if is_deep_model:
                new_key = "final_norm.weight"
            else:
                new_key = "output_norm.weight"
        elif key == "lm_head.weight": new_key = "out_head.weight"

        adapted_state_dict[new_key] = value.to(dtype) # Convert dtype during adaptation
        keys_adapted += 1
        if keys_adapted % 500 == 0:
            print(f"[DEBUG] Adapted {keys_adapted}/{len(state_dict)} keys in {time.time() - start_time:.2f}s...")

    print(f"[DEBUG] Key adaptation complete in {time.time() - start_time:.2f}s. Loading state dict into model...")
    # Load the state dict
    load_start = time.time()
    load_result = model.load_state_dict(adapted_state_dict, strict=False)
    print(f"[DEBUG] State dict loaded in {time.time() - load_start:.2f}s")

    # Filter Hopfield-related keys from missing_keys to avoid warnings for expected missing keys
    hopfield_prefixes = [
        'pre_hopfield_memory.', 'middle_hopfield_memory.', 'post_hopfield_memory.',
        'trf_blocks.', 'hopfield_memory.', 'hopfield_gate.', 'hopfield_update.', 'hopfield_combine.'
    ]

    # Special handling for trf_blocks hopfield_memory keys
    filtered_missing = []
    for k in load_result.missing_keys:
        # Skip all hopfield_memory related keys in trf_blocks
        if 'hopfield_memory' in k:
            continue
        # Skip other Hopfield-related keys
        if any(k.startswith(prefix) for prefix in hopfield_prefixes):
            continue
        # Special handling for out_head.weight, which is tied to tok_emb.weight
        if k == 'out_head.weight':
            continue
        filtered_missing.append(k)

    unexpected = load_result.unexpected_keys

    print(f"[DEBUG] Missing non-Hopfield keys: {len(filtered_missing)}")
    print(f"[DEBUG] Unexpected keys: {len(unexpected)}")

    if filtered_missing: warnings.warn(f"Weight loading missing non-Hopfield keys: {filtered_missing}")
    if unexpected: warnings.warn(f"Weight loading unexpected keys: {unexpected}")
    print("[DEBUG] Base model weights loaded into structure.")

    # Tie weights if necessary (standard for Llama)
    if "out_head.weight" not in adapted_state_dict or not torch.equal(model.out_head.weight, state_dict.get("lm_head.weight", torch.tensor([]))):
        print("[DEBUG] Applying weight tying for output head (out_head.weight tied to tok_emb.weight).")
        model.out_head.weight = model.tok_emb.weight

    # Initialize Hopfield memory AFTER embeddings are loaded
    print("[DEBUG] Starting Hopfield memory initialization...")
    init_start = time.time()
    model.initialize_hopfield_memory()
    print(f"[DEBUG] Hopfield memory initialization complete in {time.time() - init_start:.2f}s.")

    # --- SIMPLER APPROACH: No selective freezing ---
    # 1. Freeze all Llama parameters
    print("\n=== Preparing for training approach: LoRA for Llama + full training for Hopfield ===")
    print("Freezing base Llama parameters...")
    llama_param_count = 0

    for name, param in model.named_parameters():
        # Keep Hopfield params trainable
        if any(hopfield_term in name.lower() for hopfield_term in
               ['hopfield_memory', 'pre_hopfield', 'middle_hopfield', 'post_hopfield', 'hopfield_gate', 'hopfield_update', 'hopfield_combine']):
            param.requires_grad = True
        # Freeze all other (Llama) params - they'll be trained via LoRA
        else:
            param.requires_grad = False
            llama_param_count += param.numel()

    print(f"Frozen {llama_param_count:,} base Llama parameters that will be adapted via LoRA")

    # Count and examine Hopfield parameters that will be trained
    hopfield_param_count = 0
    trainable_param_count = 0
    print("\n===== TRAINABLE HOPFIELD PARAMETERS =====")
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable_param_count += p.numel()
            if 'hopfield' in name.lower():
                param_size = p.numel()
                hopfield_param_count += param_size
                print(f"  {name}: shape={p.shape}, size={param_size}")

    print(f"Total trainable Hopfield parameters: {hopfield_param_count:,}")
    print(f"Total trainable parameters before LoRA: {trainable_param_count:,}")
    print("========================================\n")

    # Apply LoRA AFTER freezing base Llama weights
    if use_lora:
        print("Applying LoRA adapters to frozen Llama parameters...")
        if lora_config_dict is None:
            # Create LoRA config from the main config
            lora_config_dict = {
                "r": config.get("lora_r", 32),
                "lora_alpha": config.get("lora_alpha", 64),
                "lora_dropout": config.get("lora_dropout", 0.05),
                "bias": "none",
                "task_type": "CAUSAL_LM"
            }

            # Extract target modules
            lora_config_dict["target_modules"] = config.get("lora_target_modules",
                                                            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

        peft_config = LoraConfig(**lora_config_dict) # Create config from dict

        # --- Important: Map target modules to scratch names ---
        scratch_target_modules = set()
        hf_target_modules = set(peft_config.target_modules) # Get target names from config
        for i in range(config.n_layers):
            hf_to_scratch_attn = {"q_proj": f"trf_blocks.{i}.att.q_proj", "k_proj": f"trf_blocks.{i}.att.k_proj", "v_proj": f"trf_blocks.{i}.att.v_proj", "o_proj": f"trf_blocks.{i}.att.o_proj"}
            hf_to_scratch_ffn = {"gate_proj": f"trf_blocks.{i}.ff.fc1", "up_proj": f"trf_blocks.{i}.ff.fc2", "down_proj": f"trf_blocks.{i}.ff.fc3"}
            for hf_name, scratch_name in {**hf_to_scratch_attn, **hf_to_scratch_ffn}.items():
                if hf_name in hf_target_modules:
                    scratch_target_modules.add(scratch_name)

        peft_config.target_modules = list(scratch_target_modules)
        print(f"Applying LoRA to modules: {peft_config.target_modules}")

        # Apply PEFT to the model
        model = get_peft_model(model, peft_config)

        # Double-check that Hopfield parameters remain trainable after LoRA application
        hopfield_trainable_count = 0
        print("\n===== VERIFYING PARAMETER STATUS AFTER LORA =====")
        for name, param in model.named_parameters():
            if any(hopfield_term in name.lower() for hopfield_term in
                   ['hopfield_memory', 'pre_hopfield', 'middle_hopfield', 'post_hopfield', 'hopfield_gate', 'hopfield_update', 'hopfield_combine']):
                if not param.requires_grad:
                    print(f"WARNING: {name} should be trainable but isn't! Fixing...")
                    param.requires_grad = True
                param_count = param.numel()
                hopfield_trainable_count += param_count

        print(f"Confirmed {hopfield_trainable_count:,} Hopfield parameters are trainable")

        # Print trainable parameter report from PEFT
        print("\n===== PEFT TRAINABLE PARAMETER REPORT =====")
        model.print_trainable_parameters()
        print("===========================================\n")

    # --- NEW: Replace model config with standard HF config for generate() compatibility ---
    print(f"[DEBUG] Replacing model's internal config object for generation compatibility...")
    from transformers import AutoConfig
    try:
        # Use model_name from the original config_ns
        hf_config = AutoConfig.from_pretrained(
            config.model_name,
            cache_dir=config.get("data_cache_dir"),
            trust_remote_code=True
        )

        # Copy essential attributes from the original ConfigNamespace to the HF config
        # These might be needed by the model's forward pass or other methods after init
        attrs_to_copy = ['gradient_checkpointing', 'hopfield_layer_placement', 'pre_middle_split', 'dtype'] # Add other necessary attrs if found
        for attr in attrs_to_copy:
            if hasattr(config, attr):
                setattr(hf_config, attr, getattr(config, attr))
                print(f"  Copied attribute '{attr}' to HF config.")
            # else:
            #     print(f"  Attribute '{attr}' not found in original config, skipping copy.")


        # Target the actual underlying model instance (handle PEFT wrapper)
        target_model_for_config_swap = model.base_model if hasattr(model, 'base_model') else model

        # Perform the replacement
        target_model_for_config_swap.config = hf_config
        print(f"  Successfully replaced internal config on '{type(target_model_for_config_swap).__name__}' with '{type(hf_config).__name__}'.")

        # Optional: Verify the replacement
        if hasattr(target_model_for_config_swap, 'config') and isinstance(target_model_for_config_swap.config, AutoConfig):
            print("  Verification successful: Model config is now a standard HF AutoConfig.")
            # Check if a copied attribute exists
            if hasattr(target_model_for_config_swap.config, 'gradient_checkpointing'):
                print(f"  Verified copied attribute 'gradient_checkpointing': {target_model_for_config_swap.config.gradient_checkpointing}")
        else:
            print("  Verification WARNING: Model config does not appear to be a standard HF AutoConfig after replacement.")

    except Exception as e:
        print(f"  WARNING: Failed to replace model config with standard HF config: {e}")
        print(f"  Generation might still fail if the custom config ({type(config).__name__}) is incompatible.")
    # --- END CONFIG REPLACEMENT ---


    print(f"Model creation and weight loading complete in {time.time() - start_time:.2f}s")
    # Return the model and the ORIGINAL ConfigNamespace (needed for optimizer setup etc. in train.py)
    return model, config

def setup_optimizer(model, config):
    """Set up optimizer with proper parameter groups and learning rates."""
    print("\n--- Optimizer Parameter Group Configuration ---")

    # Import 8-bit Adam from bitsandbytes
    try:
        from bitsandbytes.optim import Adam8bit
        has_adam8bit = True
        print("Successfully imported Adam8bit from bitsandbytes.")
    except ImportError:
        has_adam8bit = False
        print("WARNING: bitsandbytes not installed. Please install with 'pip install bitsandbytes'.")
        print("Falling back to standard AdamW optimizer (higher memory usage).")
        from torch.optim import AdamW

    # Memory optimization: Use parameter grouping more efficiently
    # Group parameters by learning rate only, not by name
    params_with_hopfield_lr = []
    params_with_base_lr = []
    hopfield_param_count = 0
    other_param_count = 0

    # Determine learning rates
    base_lr = config.get('learning_rate', 3e-5)
    hopfield_lr_multiplier = config.get('hopfield_lr_multiplier', 5.0)
    hopfield_lr = base_lr * hopfield_lr_multiplier

    # Apply batch size scaling to learning rate if specified in config
    if config.get('apply_batch_size_lr_scaling', True) and config.get('per_device_train_batch_size', 1) > 1:
        batch_size = config.get('per_device_train_batch_size', 1)
        scale_factor = (batch_size / 1) ** 0.5  # Square root scaling
        base_lr = base_lr * scale_factor
        hopfield_lr = hopfield_lr * scale_factor
        print(f"  Applied batch size scaling: LR multiplied by sqrt({batch_size}) = {scale_factor:.2f}")
        print(f"  New base LR: {base_lr:.2e}, New Hopfield LR: {hopfield_lr:.2e}")

    # Categorize parameters - simpler grouping to reduce overhead
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if this is a Hopfield parameter
        if any(hopfield_term in name.lower() for hopfield_term in
               ['hopfield_memory', 'pre_hopfield', 'middle_hopfield', 'post_hopfield', 'hopfield_gate', 'hopfield_update', 'hopfield_combine']):
            params_with_hopfield_lr.append(param)
            hopfield_param_count += param.numel()
        # All other trainable parameters
        else:
            params_with_base_lr.append(param)
            other_param_count += param.numel()

    # Print debug information
    print(f"  Trainable parameter distribution:")
    print(f"  - Hopfield parameters: {len(params_with_hopfield_lr)} parameters ({hopfield_param_count:,} elements)")
    print(f"  - Other trainable parameters: {len(params_with_base_lr)} parameters ({other_param_count:,} elements)")
    print(f"  - Total trainable parameters: {hopfield_param_count + other_param_count:,} elements")

    # Create optimizer with parameter groups - simpler grouping
    optimizer_groups = []

    # Add Hopfield parameters with their special learning rate
    if params_with_hopfield_lr:
        optimizer_groups.append({
            'params': params_with_hopfield_lr,
            'lr': hopfield_lr
        })

    # Add all other trainable parameters with base learning rate
    if params_with_base_lr:
        optimizer_groups.append({
            'params': params_with_base_lr,
            'lr': base_lr
        })

    # Memory optimization: Use 8-bit optimizer with lower memory settings
    betas = (config.get('adam_beta1', 0.9), config.get('adam_beta2', 0.999))
    eps = config.get('adam_epsilon', 1e-8)
    weight_decay = config.get('weight_decay', 0.0)

    print(f"  Using {'8-bit' if has_adam8bit else 'standard'} Adam optimizer with betas={betas}, eps={eps}")

    if has_adam8bit:
        optimizer = Adam8bit(
            optimizer_groups,
            lr=base_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        print("  Using 8-bit quantization for optimizer states (significant memory savings).")
    else:
        optimizer = AdamW(
            optimizer_groups,
            lr=base_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        print("  WARNING: Using standard AdamW (higher memory usage).")

    print(f"  Optimizer initialized with {len(optimizer_groups)} parameter groups.")

    # Print memory savings estimate
    total_params = hopfield_param_count + other_param_count
    standard_mem = total_params * 2 * 4 / (1024**3)  # 2 states, 4 bytes each (FP32), in GB
    quantized_mem = total_params * 2 * 1 / (1024**3)  # 2 states, 1 byte each (INT8), in GB
    print(f"  Estimated optimizer state memory: {quantized_mem if has_adam8bit else standard_mem:.2f} GB")
    if has_adam8bit:
        print(f"  Estimated memory savings vs. standard Adam: {standard_mem - quantized_mem:.2f} GB")

    return optimizer

class DeepHopfieldTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print("[DEBUG DHBlock] Starting initialization...")
        # Regular transformer components
        self.norm1 = RMSNorm(cfg.emb_dim, eps=cfg.get("rms_norm_eps", 1e-5), dtype=cfg.dtype)
        self.att = GroupedQueryAttention(cfg)
        self.norm2 = RMSNorm(cfg.emb_dim, eps=cfg.get("rms_norm_eps", 1e-5), dtype=cfg.dtype)
        self.ff = FeedForward(cfg)
        # Hopfield Memory
        print("[DEBUG DHBlock] Creating Hopfield memory layer...")
        self.hopfield_memory = HopfieldMemoryLayer(cfg)

        # Optimization: Add option to skip memory updates during forward pass
        # This allows explicit control over when memory updates happen
        self.skip_memory_update = False

        # Store config for later reference
        self.config = cfg

        print("[DEBUG DHBlock] Initialization complete")

    def reset_memory(self):
        """Explicitly reset the Hopfield memory state to prevent gradient chain buildup."""
        if hasattr(self, 'hopfield_memory') and hasattr(self.hopfield_memory, 'reset_memory'):
            self.hopfield_memory.reset_memory()
        else:
            print("Warning: Could not reset Hopfield memory in transformer block - missing attribute")

    def forward(self, x, position_ids=None):
        # Optimization: Reuse tensor allocations where possible
        # and avoid unnecessary copies

        # Pre-norm architecture: norm->hopfield->attn->residual->norm->ffn->residual

        # Apply layernorm (no need to clone x here)
        hidden_states_norm = self.norm1(x)

        # Process through Hopfield layer - avoid making copies
        hidden_states_hopfield = self.hopfield_memory(hidden_states_norm)

        # Update memory state if needed
        if (not self.skip_memory_update and
                hasattr(self.hopfield_memory, 'update_strategy') and
                self.hopfield_memory.update_strategy != "none" and
                self.training):
            self.hopfield_memory.update_memory()

        # Apply attention directly to Hopfield output to avoid creating an extra tensor
        attn_output = self.att(hidden_states_hopfield, position_ids=position_ids)

        # First residual connection - reuse x if possible
        if x.requires_grad:
            # Cannot modify in-place if we need gradients
            h = x + attn_output
        else:
            # Use in-place addition to save memory when possible
            h = x.clone()
            h.add_(attn_output)

        # Second layernorm directly on h
        hidden_states_norm2 = self.norm2(h)

        # Feed-forward network
        ff_output = self.ff(hidden_states_norm2)

        # Second residual connection - reuse h if possible
        if h.requires_grad:
            out = h + ff_output
        else:
            # Use in-place addition when possible
            out = h
            out.add_(ff_output)

        return out

    def set_skip_memory_update(self, skip=True):
        """Set whether to skip memory updates during forward pass."""
        self.skip_memory_update = skip
        return self

class DeepHopfieldLlama3Model(nn.Module, GenerationMixin):
    main_input_name = "input_ids"

    def __init__(self, cfg):
        super().__init__()
        print("[DEBUG DeepHopfield] Starting initialization...")
        # Ensure cfg is a ConfigNamespace
        if isinstance(cfg, dict):
            cfg = ConfigNamespace(**cfg)
        self.cfg = cfg # Keep original ConfigNamespace if needed elsewhere

        # Set model_type if not present
        if not hasattr(cfg, 'model_type'):
            cfg.model_type = 'deep_hopfield_llama'

        self.config = cfg # Use the ConfigNamespace directly
        self.config.is_encoder_decoder = False # Explicitly set for GenerationMixin compatibility

        # Set use_cache on the config object (standard)
        self.config.use_cache = False
        # Set _supports_cache_class directly on the model instance
        self._supports_cache_class = False

        # Create embedding and token types
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.emb_dim, dtype=self.config.dtype)

        # Create transformer blocks with integrated hopfield memory
        print("[DEBUG DeepHopfield] Creating transformer blocks with integrated Hopfield memory...")
        self.trf_blocks = nn.ModuleList([DeepHopfieldTransformerBlock(self.config) for _ in range(self.config.n_layers)])

        # Create final normalization layer and output head
        self.final_norm = RMSNorm(self.config.emb_dim, eps=self.config.get("rms_norm_eps", 1e-5), dtype=self.config.dtype)
        self.out_head = nn.Linear(self.config.emb_dim, self.config.vocab_size, bias=False, dtype=self.config.dtype)

        # Tie weights
        self.out_head.weight = self.tok_emb.weight

        # Batch optimization flag - if True, memory updates are deferred until explicitly triggered
        self.batch_memory_optimization = False

        # Enable gradient checkpointing by default for better memory efficiency
        self.gradient_checkpointing_enable()

        print("[DEBUG DeepHopfield] Model initialization complete")

    def enable_batch_memory_optimization(self, enable=True):
        """Enable/disable batch-level memory update optimization.

        When enabled, Hopfield memory updates are delayed until explicitly triggered
        by calling update_all_memories(). This improves efficiency when processing
        multiple sequences from the same document in separate forward passes.
        """
        self.batch_memory_optimization = enable
        # Set skip_memory_update flag on all transformer blocks
        for block in self.trf_blocks:
            if hasattr(block, 'set_skip_memory_update'):
                block.set_skip_memory_update(enable)

        print(f"[DEBUG] {'Enabled' if enable else 'Disabled'} batch memory optimization. "
              f"Memory updates will {'be deferred' if enable else 'happen during forward pass'}.")
        return self

    def update_all_memories(self):
        """Explicitly update Hopfield memories across all transformer blocks."""
        if not self.batch_memory_optimization:
            print("[WARNING] Called update_all_memories() but batch_memory_optimization is disabled.")
            return self

        update_count = 0
        for i, block in enumerate(self.trf_blocks):
            if (hasattr(block, 'hopfield_memory') and
                    hasattr(block.hopfield_memory, 'update_memory') and
                    block.hopfield_memory.update_strategy != "none"):
                block.hopfield_memory.update_memory()
                update_count += 1

        print(f"[DEBUG] Updated {update_count}/{len(self.trf_blocks)} Hopfield memories explicitly.")
        return self

    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings and output projection layer."""
        old_embeddings = self.tok_emb
        old_num_tokens = old_embeddings.weight.size(0)

        # Create new embedding layer
        self.tok_emb = nn.Embedding(new_num_tokens, self.config.emb_dim, dtype=self.config.dtype)

        # Copy weights from old embeddings
        if old_embeddings is not None and old_num_tokens > 0:
            self.tok_emb.weight.data[:old_num_tokens] = old_embeddings.weight.data

        # Resize output head
        old_output = self.out_head
        self.out_head = nn.Linear(self.config.emb_dim, new_num_tokens, bias=False, dtype=self.config.dtype)

        # Copy weights for output head
        if old_output is not None and old_num_tokens > 0:
            self.out_head.weight.data[:old_num_tokens] = old_output.weight.data

        # Update config
        self.cfg['vocab_size'] = new_num_tokens
        self.config.vocab_size = new_num_tokens

        return self

    @property
    def device(self) -> torch.device:
        # Return the device of the first parameter found
        try:
            return next(self.parameters()).device
        except StopIteration:
            # Handle case where model has no parameters or is on meta device
            # Try checking buffers if no parameters
            try:
                return next(self.buffers()).device
            except StopIteration:
                # Default to CPU if no parameters or buffers found
                return torch.device("cpu")

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.Tensor]] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs):
        # Memory optimization: avoid redundant condition checks and tensor creations

        # Quick validation of inputs
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Setup output flags
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", False)
        output_attentions = output_attentions if output_attentions is not None else getattr(self.config, "output_attentions", False)
        output_hidden_states = output_hidden_states if output_hidden_states is not None else getattr(self.config, "output_hidden_states", False)
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", True)

        # Get device once
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Generate position IDs if not provided (only once)
        if position_ids is None and input_ids is not None:
            position_ids = torch.arange(input_ids.size(1), device=device).unsqueeze(0)

        # Embedding lookup
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.tok_emb(input_ids)

        # Pass through Transformer blocks with memory optimization
        all_hidden_states = () if output_hidden_states else None

        # Process through transformer blocks with gradient checkpointing
        for i, block in enumerate(self.trf_blocks):
            if output_hidden_states:
                all_hidden_states += (h,)

            if self.config.gradient_checkpointing and self.training:
                # Use torch.utils.checkpoint to save memory during training
                # Disable reentrant to save more memory at the cost of slightly slower backward
                h = torch.utils.checkpoint.checkpoint(
                    block, h, position_ids, use_reentrant=False
                )
            else:
                h = block(h, position_ids=position_ids)

        # Store final hidden state if needed
        if output_hidden_states:
            all_hidden_states += (h,)

        # Apply final layer norm and compute logits
        h_final = self.final_norm(h)
        logits = self.out_head(h_final)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

            # Use standard cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

        # Return in the format requested
        if not return_dict:
            outputs = (logits,) + (None,) + (all_hidden_states,) + (None,)
            return (loss,) + outputs if loss is not None else outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.config.gradient_checkpointing = True
        # Use higher precision for checkpoint for numerical stability
        torch.utils.checkpoint.use_reentrant = False
        print("Gradient checkpointing enabled with use_reentrant=False for better memory efficiency.")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.config.gradient_checkpointing = False
        print("Gradient checkpointing disabled.")

    def initialize_hopfield_memory(self):
        """Initialize Hopfield memory layers in each transformer block."""
        print("[DEBUG DeepHopfield] Starting Hopfield memory initialization in all blocks...")
        if hasattr(self, 'trf_blocks'):
            for i, block in enumerate(self.trf_blocks):
                print(f"[DEBUG DeepHopfield] Initializing Hopfield memory in block {i}/{len(self.trf_blocks)}...")
                if hasattr(block, 'hopfield_memory') and hasattr(block.hopfield_memory, '_initialized'):
                    if not block.hopfield_memory._initialized:
                        print(f"[DEBUG DeepHopfield] Block {i} needs initialization...")
                        block.hopfield_memory.initialize_memory(self.tok_emb.weight)
                    else:
                        print(f"[DEBUG DeepHopfield] Block {i} already initialized, skipping...")
                else:
                    print(f"[DEBUG DeepHopfield] Block {i} doesn't have proper Hopfield attributes!")
            print("[DEBUG DeepHopfield] Hopfield memory initialization complete for all blocks")
        else:
            print("[DEBUG DeepHopfield] No transformer blocks found!")

    def reset_all_memories(self):
        """Reset all Hopfield memory layers to prevent computational history leaking between examples."""
        print("[DEBUG DeepHopfield] Resetting all Hopfield memory layers...")
        if hasattr(self, 'trf_blocks'):
            for i, block in enumerate(self.trf_blocks):
                if hasattr(block, 'reset_memory'):
                    block.reset_memory()
                elif hasattr(block, 'hopfield_memory') and hasattr(block.hopfield_memory, 'reset_memory'):
                    block.hopfield_memory.reset_memory()
            print(f"[DEBUG DeepHopfield] Reset {len(self.trf_blocks)} transformer blocks")
        else:
            print("[DEBUG DeepHopfield] No transformer blocks found to reset!")

        # Return self for method chaining
        return self

    def can_generate(self) -> bool:
        """Check if the model can generate text (essential for GenerationMixin)."""
        # Check if the model has the necessary attributes for generation
        # Usually checks for an LM head and the main input name in config
        config = getattr(self, "config", None)
        lm_head_present = isinstance(getattr(self, "out_head", None), nn.Linear)
        main_input_in_config = hasattr(config, self.main_input_name)

        # return lm_head_present and main_input_in_config and hasattr(self.config, "vocab_size")
        # Simpler check: Assume if it has the head and main input name, it can generate
        return lm_head_present and hasattr(self, self.main_input_name)

    def _validate_model_class(self):
        """
        Override the GenerationMixin validation check.
        Our custom model doesn't follow standard HF naming conventions,
        but we've implemented the necessary components for generation.
        """
        pass # Bypass the check
