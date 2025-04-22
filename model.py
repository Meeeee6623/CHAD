import torch
import torch.nn as nn
import math
import warnings
from typing import Optional, Tuple, List
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
import types

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

        # Reshape x to [B, T, H, Dh] before compute_rope which expects [B, H, T, Dh]
        # This is awkward. Let's adjust compute_rope or apply RoPE here directly.
        # Applying RoPE directly to [B, T, H, Dh]
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

        # 6. Transpose K for matmul
        keys_t = keys.transpose(-2, -1) # [B, H, Dh, T]

        # === DEBUG PRINTS ===
        # print(f"[DEBUG] queries shape: {queries.shape}")
        # print(f"[DEBUG] keys_t shape: {keys_t.shape}")
        # === END DEBUG PRINTS ===

        # 7. Attention calculation
        attn_scores = queries @ keys_t

        # 8. Apply causal mask
        mask_bool = self.mask[:num_tokens, :num_tokens] # Slice mask [T, T]
        # Expand mask to attention score shape [B, H, Tq, Tk]
        # No batch or head dim needed for broadcast if mask is [Tq, Tk]
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1)
        attn_weights = attn_weights.to(values.dtype)

        # 9. Compute context vector
        context_vec = (attn_weights @ values).transpose(1, 2) # [B, T, H, Dh]
        context_vec = context_vec.reshape(b, num_tokens, self.num_heads * self.head_dim) # [B, T, E]
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
    # (Use the refined HopfieldMemoryLayer class from previous response)
    # Ensure RMSNorm is used if base uses it
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg # Store config SimpleNamespace directly
        self.emb_dim = cfg.emb_dim
        self.n_heads = cfg.hopfield_heads
        self.n_memory_slots = cfg.hopfield_memory_slots
        self.pattern_dim = cfg.emb_dim
        self.head_dim = self.pattern_dim // self.n_heads
        if self.head_dim * self.n_heads != self.pattern_dim:
             raise ValueError("emb_dim must be divisible by hopfield_heads")

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

        self.storedpatterns = nn.Parameter(
            torch.empty(self.n_heads, self.n_memory_slots, self.head_dim, dtype=self.dtype)
        )
        self._initialized = False

        self.q_proj = nn.Linear(self.emb_dim, self.pattern_dim, bias=False, dtype=self.dtype)
        self.k_proj = nn.Linear(self.head_dim, self.head_dim, bias=False, dtype=self.dtype)
        self.v_proj = nn.Linear(self.head_dim, self.head_dim, bias=False, dtype=self.dtype)
        # Correct beta shape for broadcasting: [1, H_hop, 1, 1]
        self.beta = nn.Parameter(torch.ones(1, self.n_heads, 1, 1, dtype=self.dtype))

        # Match Norm type with base model
        self.norm_query = RMSNorm(self.emb_dim, eps=cfg.get("rms_norm_eps", 1e-5), dtype=self.dtype)
        self.norm_retrieved = RMSNorm(self.emb_dim, eps=cfg.get("rms_norm_eps", 1e-5), dtype=self.dtype)


        if self.update_strategy == "gated":
            gate_input_dim = self.emb_dim * 2
            gate_output_dim = self.n_heads * self.n_memory_slots * self.head_dim
            self.gate_linear = nn.Linear(gate_input_dim, gate_output_dim, bias=False, dtype=self.dtype)
            print(f"Initialized Gated Update mechanism.")

        if self.combine_method == "concat":
            self.combine_proj = nn.Linear(self.emb_dim * 2, self.emb_dim, bias=False, dtype=self.dtype)
            print("Initialized Concat combine method.")

        self._clear_last_state()

    def initialize_memory(self, embedding_matrix: Optional[torch.Tensor] = None):
        # (Same as previous implementation)
        if self._initialized: return
        print(f"Initializing Hopfield memory ({self.init_method})...")
        with torch.no_grad():
            if self.init_method == "embedding_sampling" and embedding_matrix is not None:
                vocab_size, emb_dim_vocab = embedding_matrix.shape
                if emb_dim_vocab != self.emb_dim:
                    warnings.warn(f"Vocab E({emb_dim_vocab}) != Hopfield E({self.emb_dim}). Falling back to random init.")
                    self.storedpatterns.data.normal_(mean=0.0, std=0.02)
                else:
                    num_samples = self.n_heads * self.n_memory_slots
                    indices = torch.randint(0, vocab_size, (num_samples,), device=embedding_matrix.device)
                    sampled_embeddings = embedding_matrix[indices].to(device=self.storedpatterns.device, dtype=self.dtype)
                    try:
                         # Ensure correct reshaping: Sample E, take first Dh, reshape to [H, S, Dh]
                         target_shape = (self.n_heads, self.n_memory_slots, self.head_dim)
                         # Slice the sampled embeddings to match the head dimension
                         sampled_head_dim_embeddings = sampled_embeddings[:, :self.head_dim]
                         self.storedpatterns.data.copy_(
                              sampled_head_dim_embeddings.view(target_shape)
                         )
                         print(f"Initialized stored patterns by sampling embeddings (sliced to head_dim).")
                    except RuntimeError as e:
                         warnings.warn(f"Error assigning sampled embeddings (shape mismatch?): {e}. Falling back to random.")
                         self.storedpatterns.data.normal_(mean=0.0, std=0.02)
            else:
                 if self.init_method == "embedding_sampling":
                     warnings.warn("Embedding matrix not provided for 'embedding_sampling' init. Using random.")
                 self.storedpatterns.data.normal_(mean=0.0, std=0.02)
        self._initialized = True


    def forward(self, query_input): # Removed mask pass-through for simplicity
        if not self._initialized: raise RuntimeError("Hopfield memory not initialized.")

        batch_size, seq_len, _ = query_input.shape
        query_input_dtype = query_input.dtype
        query_input = query_input.to(self.dtype)
        device = query_input.device

        query_norm = self.norm_query(query_input)
        query = self.q_proj(query_norm)
        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        storedpatterns_device = self.storedpatterns.to(device=device, dtype=self.dtype)
        keys_flat = self.k_proj(storedpatterns_device.view(-1, self.head_dim))
        values_flat = self.v_proj(storedpatterns_device.view(-1, self.head_dim))
        keys = keys_flat.view(1, self.n_heads, self.n_memory_slots, self.head_dim)
        values = values_flat.view(1, self.n_heads, self.n_memory_slots, self.head_dim)

        current_query = query
        if self.clamp_beta is not None:
            with torch.no_grad(): self.beta.data.clamp_(min=1e-2, max=self.clamp_beta) # Use .data for in-place clamp

        attn_scores = torch.matmul(current_query, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores * self.beta.to(device)
        attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(self.dtype)
        retrieved_memory = torch.matmul(attn_weights, values)

        # === DEBUG Hopfield ===
        # print(f"[DEBUG Hopfield] retrieved_memory shape: {retrieved_memory.shape}")
        retrieved_memory_t = retrieved_memory.transpose(1, 2)
        # print(f"[DEBUG Hopfield] retrieved_memory_t shape: {retrieved_memory_t.shape}")
        # === END DEBUG Hopfield ===

        # Ensure contiguous before reshape
        final_retrieved_reshaped = retrieved_memory_t.contiguous().reshape(batch_size, seq_len, self.pattern_dim)
        final_retrieved_norm = self.norm_retrieved(final_retrieved_reshaped)

        if self.training or self.update_strategy != "none":
            self._last_attn_weights = attn_weights.detach()
            self._last_query_input = query_input.detach()
            self._last_query_proj = query.detach()
            self._last_retrieved_memory_norm = final_retrieved_norm.detach()

        if self.combine_method == "concat":
            combined = torch.cat((query_input, final_retrieved_norm), dim=-1)
            output = self.combine_proj(combined.to(self.combine_proj.weight.dtype))
        else:
            if self.combine_method != "add": warnings.warn(f"Unknown combine: {self.combine_method}, using add.")
            output = query_input + final_retrieved_norm

        return output.to(query_input_dtype)

    def update_memory(self):
        # (Use the update_memory logic from the previous response)
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
                     avg_query = torch.mean(self._last_query_input.to(device, dtype), dim=(0, 1))
                     try:
                          target_per_head = avg_query.view(self.n_heads, self.head_dim)
                          update_target = target_per_head.unsqueeze(1).expand_as(self.storedpatterns)
                     except RuntimeError as e: print(f"Error reshaping avg_query: {e}")
                else: print("Warning: Missing query input for update target.")
            # Add other methods ('avg_retrieved', 'attention_weighted_query') here if implementing
            elif self.update_target_method == "avg_retrieved":
                 if self._last_retrieved_memory_norm is not None:
                      avg_retrieved = torch.mean(self._last_retrieved_memory_norm.to(device, dtype), dim=(0, 1))
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
                    if self.gate_input_pooling == "mean":
                        query_pooled = torch.mean(self._last_query_input.to(device), dim=(0, 1))
                        retrieved_pooled = torch.mean(self._last_retrieved_memory_norm.to(device), dim=(0, 1))
                    elif self.gate_input_pooling == "max":
                        query_pooled = torch.max(self._last_query_input.to(device), dim=1)[0].mean(dim=0)
                        retrieved_pooled = torch.max(self._last_retrieved_memory_norm.to(device), dim=1)[0].mean(dim=0)

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
        """Clears the internal state used for memory updates."""
        self._clear_last_state()
        # Optional: Reset any other state if needed (e.g., accumulated statistics)
        print("Hopfield memory state cleared.")


# --- HAT Model (Using scratch structure) ---
class HopfieldLlama3Model(nn.Module, GenerationMixin):
    main_input_name = "input_ids"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg # Keep original dict if needed elsewhere
        cfg.setdefault('model_type', 'custom_llama')
        self.config = ConfigNamespace(**cfg) # Config object used by GenerationMixin
        self.config.is_encoder_decoder = False # Explicitly set for GenerationMixin compatibility

        # --- REVISED APPROACH ---
        # Set use_cache on the config object (standard)
        self.config.use_cache = False
        # Set _supports_cache_class directly on the model instance
        self._supports_cache_class = False # <<< Set directly on self
        # --- END REVISION ---

        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.emb_dim, dtype=self.config.dtype)
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.config.n_layers)]
        )
        self.hopfield_memory = HopfieldMemoryLayer(self.config)
        self.final_norm = RMSNorm(self.config.emb_dim, eps=self.config.get("rms_norm_eps", 1e-5), dtype=self.config.dtype)
        self.out_head = nn.Linear(self.config.emb_dim, self.config.vocab_size, bias=False, dtype=self.config.dtype)

        if self.hopfield_memory.init_method != "embedding_sampling":
             self.hopfield_memory.initialize_memory(None)

        if self.config.get("gradient_checkpointing", False):
             self.gradient_checkpointing_enable()
             
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
                past_key_values: Optional[List[torch.Tensor]] = None, # Still here, though unused by our model
                use_cache: Optional[bool] = None, # Still here, though unused by our model
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs # <<< ADD THIS to accept and ignore extra arguments
               ):
        # input_ids: [B, T]
        # attention_mask: [B, T]
        # inputs_embeds: [B, T, E]

        # Determine whether to output attentions/hidden states based on args or config
        # Note: We should check the passed `use_cache` argument if we were implementing cache
        use_cache = use_cache if use_cache is not None else self.config.get("use_cache", False) # Respect passed argument if available
        output_attentions = output_attentions if output_attentions is not None else self.config.get("output_attentions", False)
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.get("output_hidden_states", False)
        return_dict = return_dict if return_dict is not None else self.config.get("use_return_dict", True) # HF usually defaults use_return_dict to True

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        if inputs_embeds is not None:
            # This model currently relies on input_ids for embedding lookup.
            # If inputs_embeds is needed, the embedding step needs modification.
            warnings.warn("inputs_embeds provided but HopfieldLlama3Model currently uses input_ids for embedding lookup.")
            # For now, try to proceed assuming input_ids were intended or derived elsewhere if possible
            # Get batch_size and seq_len from inputs_embeds if input_ids is None
            batch_size, seq_len, _ = inputs_embeds.shape
            # Need to handle device placement if inputs_embeds is primary
            device = inputs_embeds.device
            # Set input_ids to None explicitly if embeds are primary?
            # input_ids = None # This might break downstream logic if embeds aren't used
        elif input_ids is not None:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
        else:
            # Should be unreachable due to earlier check
            raise ValueError("Logic error: Need either input_ids or inputs_embeds")


        # Ensure position_ids are generated correctly based on available input
        if position_ids is None:
             past_length = 0
             if past_key_values is not None and past_key_values[0] is not None:
                  if seq_len == 1 and len(past_key_values[0]) > 0:
                       try: past_length = past_key_values[0][0].shape[2]
                       except: pass
                  position_ids = torch.arange(past_length, past_length + seq_len, device=device).unsqueeze(0)
             else:
                  position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # --- Embedding Lookup --- #
        if inputs_embeds is not None:
             h = inputs_embeds
        elif input_ids is not None:
             h = self.tok_emb(input_ids)
        else:
             raise ValueError("Logic error: Could not determine input embeddings")

        # Pass through Transformer blocks
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for i, block in enumerate(self.trf_blocks):
             if output_hidden_states:
                  all_hidden_states += (h,)

             # Handle Gradient Checkpointing
             if self.config.gradient_checkpointing and self.training:
                 # Checkpoint needs to return outputs compatible with non-checkpointed version
                 # If block returns cache/attentions, checkpoint must also
                 # *** Current TransformerBlock doesn't return attentions/cache ***
                 # *** Need to modify TransformerBlock if these outputs are needed ***
                 layer_outputs = torch.utils.checkpoint.checkpoint(
                      block, h, position_ids, use_reentrant=False # Checkpoint only passes positional args
                 )
                 h = layer_outputs
             else:
                 # *** Modify block call if it were to support cache/attentions ***
                 # layer_outputs = block(h, position_ids=position_ids, past_key_value=..., use_cache=use_cache, output_attentions=output_attentions)
                 # h = layer_outputs[0]
                 # if use_cache: next_decoder_cache += (layer_outputs[1],)
                 # if output_attentions: all_self_attns += (layer_outputs[2],)
                 h = block(h, position_ids=position_ids)

        if output_hidden_states:
             all_hidden_states += (h,)

        # Pass through Hopfield Layer
        h_hopfield = self.hopfield_memory(h)

        # Update memory AFTER Hopfield forward pass, using its stored state
        if self.training and self.hopfield_memory.update_strategy != "none":
             self.hopfield_memory.update_memory()

        h_final = self.final_norm(h_hopfield)
        logits = self.out_head(h_final)

        # --- Calculate Loss if labels provided --- #
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # Prepare output
        if not return_dict:
            # Note: Scratch model doesn't produce KV cache or block attentions currently
            outputs = (logits,) + (None,) + (all_hidden_states,) + (None,)
            return (loss,) + outputs if loss is not None else outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None, # Dummy value
            hidden_states=all_hidden_states,
            attentions=all_self_attns, # Dummy value (None)
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # This is a basic implementation. Assumes no KV cache is used by the scratch model.
        # If your model uses KV caching, this needs to be adapted.
        model_inputs = {"input_ids": input_ids}

        # Add position_ids - crucial for RoPE
        # During generation, if past_key_values exist, input_ids is usually [B, 1]
        # Position IDs should correspond to the *next* token position
        if past_key_values is not None:
             # Estimate past sequence length - requires knowing KV cache structure
             # Placeholder: Assume past_key_values is layer-based tuple/list
             try:
                  # Example: Accessing shape of key tensor in the first layer's cache
                  # Adjust indices based on your cache structure if implemented
                  past_length = past_key_values[0][0].shape[2] # [B, H, T_past, Dh]
             except (TypeError, IndexError):
                  # Fallback if cache structure is unknown or empty
                  past_length = 0
             position_ids = torch.tensor([[past_length + input_ids.shape[1] - 1]], device=input_ids.device, dtype=torch.long)
        else:
             seq_length = input_ids.shape[1]
             position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)

        model_inputs['position_ids'] = position_ids

        # Include other potential kwargs if needed by your forward pass
        model_inputs.update(kwargs)

        # Return dummy past_key_values if not implemented
        # Or return the actual cache if your forward pass generates it
        return model_inputs

    def initialize_hopfield_memory(self):
         if not self.hopfield_memory._initialized:
              print("Explicitly initializing Hopfield memory...")
              self.hopfield_memory.initialize_memory(self.tok_emb.weight)

    def gradient_checkpointing_enable(self):
         self.config.gradient_checkpointing = True
         print("Gradient checkpointing enabled for Transformer blocks.")

    def gradient_checkpointing_disable(self):
         self.config.gradient_checkpointing = False
         print("Gradient checkpointing disabled.")

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

# --- Function to create model and load weights ---
def create_model_and_load_weights(config: dict, use_lora: bool, lora_config_dict: Optional[dict] = None):
    import yaml # Keep imports local if only used here
    from safetensors.torch import load_file as load_safetensors
    from huggingface_hub import hf_hub_download
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    # REMOVED: Config loading - use the passed config dictionary directly
    # with open(config_path, 'r') as f:
    #     config = yaml.safe_load(f)

    # --- Ensure critical numeric params from config dict are floats ---
    try: config['rms_norm_eps'] = float(config.get('rms_norm_eps', 1e-5))
    except (ValueError, TypeError): raise TypeError(f"Config 'rms_norm_eps' must be a number.")
    try: config['rope_base'] = float(config.get('rope_base', 500000.0))
    except (ValueError, TypeError): raise TypeError(f"Config 'rope_base' must be a number.")
    # --- End Numeric Param Conversion ---

    # Determine torch dtype
    dtype_str = config.get("model_dtype", "float32")
    if dtype_str == "bfloat16" and torch.cuda.is_bf16_supported(): dtype = torch.bfloat16
    elif dtype_str == "float16": dtype = torch.float16
    else: dtype = torch.float32
    config["dtype"] = dtype

    print(f"Initializing HopfieldLlama3Model (Scratch Base) with dtype: {dtype}")
    # Initialize OUR model structure
    model = HopfieldLlama3Model(config)

    # Load pretrained weights from HF Hub into OUR structure
    print(f"Loading pretrained weights for {config['model_name']}...")
    try:
        # Handle sharded weights - Assume 2 shards for 3B model as a likely default
        # TODO: Ideally, determine num_shards dynamically if possible (e.g., listing files)
        num_shards = config.get("num_weight_shards", 2) # Try 2 shards by default
        print(f"Assuming {num_shards} weight shards for {config['model_name']}.")
        model_files_info = [(f"model-{i+1:05d}-of-{num_shards:05d}.safetensors" if num_shards > 1 else "model.safetensors") for i in range(num_shards)]

        state_dict = {}
        for filename in model_files_info:
             weights_file = hf_hub_download(
                 repo_id=config['model_name'],
                 filename=filename,
                 cache_dir=config.get("data_cache_dir", "./model_cache"),
                 resume_download=True,
             )
             print(f"Loading weights from {filename} to CPU...")
             shard_state_dict = load_safetensors(weights_file, device="cpu")
             state_dict.update(shard_state_dict)
             del shard_state_dict # Free memory

    except Exception as e:
         raise RuntimeError(f"Failed to load weights: {e}. Check model name and HF Hub access.")


    # Adapt HF state_dict keys to our scratch model names
    adapted_state_dict = {}
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
        elif key == "model.norm.weight": new_key = "final_norm.weight"
        elif key == "lm_head.weight": new_key = "out_head.weight"

        adapted_state_dict[new_key] = value.to(dtype) # Convert dtype during adaptation

    # Load the state dict
    load_result = model.load_state_dict(adapted_state_dict, strict=False)
    missing = [k for k in load_result.missing_keys if not k.startswith("hopfield_memory.")]
    unexpected = load_result.unexpected_keys
    if missing: warnings.warn(f"Weight loading missing non-Hopfield keys: {missing}")
    if unexpected: warnings.warn(f"Weight loading unexpected keys: {unexpected}")
    print("Base model weights loaded into HAT structure.")

    # Tie weights if necessary (standard for Llama)
    if "out_head.weight" not in adapted_state_dict or torch.equal(model.out_head.weight, state_dict.get("lm_head.weight", torch.tensor([]))): # Check if separate head was loaded
         print("Applying weight tying for output head.")
         model.out_head.weight = model.tok_emb.weight


    # Initialize Hopfield memory AFTER embeddings are loaded
    model.initialize_hopfield_memory()

    # Apply LoRA AFTER loading base weights and initializing Hopfield
    if use_lora:
        print("Applying LoRA adapters...")
        if lora_config_dict is None: raise ValueError("LoRA config dict required when use_lora=True")
        peft_config = LoraConfig(**lora_config_dict) # Create config from dict

        # --- Important: Map target modules to scratch names ---
        scratch_target_modules = set()
        hf_target_modules = set(peft_config.target_modules) # Get target names from config
        for i in range(config["n_layers"]):
            hf_to_scratch_attn = {"q_proj": f"trf_blocks.{i}.att.q_proj", "k_proj": f"trf_blocks.{i}.att.k_proj", "v_proj": f"trf_blocks.{i}.att.v_proj", "o_proj": f"trf_blocks.{i}.att.o_proj"}
            hf_to_scratch_ffn = {"gate_proj": f"trf_blocks.{i}.ff.fc1", "up_proj": f"trf_blocks.{i}.ff.fc2", "down_proj": f"trf_blocks.{i}.ff.fc3"}
            for hf_name, scratch_name in {**hf_to_scratch_attn, **hf_to_scratch_ffn}.items():
                 if hf_name in hf_target_modules:
                      scratch_target_modules.add(scratch_name)
        # Add embedding/output head if needed (less common for LoRA)
        # if "embed_tokens" in hf_target_modules: scratch_target_modules.add("tok_emb")
        # if "lm_head" in hf_target_modules: scratch_target_modules.add("out_head")

        peft_config.target_modules = list(scratch_target_modules)
        print(f"Applying LoRA to modules: {peft_config.target_modules}")

        # Apply PEFT to the model
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        # Ensure Hopfield parameters remain trainable if LoRA is applied
        for name, param in model.named_parameters():
            if 'hopfield_memory' in name:
                param.requires_grad = True


    print("Model creation and weight loading complete.")
    return model, config