import torch
import torch.nn as nn
import math
import warnings
from typing import Optional, Tuple, List
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
import types
from copy import deepcopy
import random

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
        
        # Check for essential attributes with appropriate defaults and warnings
        if not hasattr(cfg, 'emb_dim'):
            warnings.warn("ConfigNamespace missing 'emb_dim' attribute. Using default value 4096.")
            d_in = d_out = 4096
        else:
            d_in = d_out = getattr(cfg, 'emb_dim')
        
        if not hasattr(cfg, 'n_heads'):
            warnings.warn("ConfigNamespace missing 'n_heads' attribute. Using default value 32.")
            self.num_heads = 32
        else:
            self.num_heads = getattr(cfg, 'n_heads')
            
        if not hasattr(cfg, 'n_kv_groups'):
            warnings.warn("ConfigNamespace missing 'n_kv_groups' attribute. Using default value (n_heads / 4).")
            self.num_kv_groups = max(1, self.num_heads // 4)
        else:
            self.num_kv_groups = getattr(cfg, 'n_kv_groups')
            
        if not hasattr(cfg, 'dtype'):
            warnings.warn("ConfigNamespace missing 'dtype' attribute. Using default torch.float32.")
            dtype = torch.float32
        else:
            dtype = getattr(cfg, 'dtype')

        assert d_out % self.num_heads == 0, f"emb_dim {d_out} must be divisible by n_heads {self.num_heads}"
        assert self.num_heads % self.num_kv_groups == 0, f"n_heads {self.num_heads} must be divisible by n_kv_groups {self.num_kv_groups}"

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
        
        # Check for essential attributes with appropriate defaults and warnings
        if not hasattr(cfg, 'emb_dim'):
            warnings.warn("ConfigNamespace missing 'emb_dim' attribute. Using default value 4096.")
            emb_dim = 4096
        else:
            emb_dim = getattr(cfg, 'emb_dim')
            
        if not hasattr(cfg, 'hidden_dim'):
            warnings.warn("ConfigNamespace missing 'hidden_dim' attribute. Using default value (emb_dim * 4).")
            hidden_dim = emb_dim * 4
        else:
            hidden_dim = getattr(cfg, 'hidden_dim')
            
        if not hasattr(cfg, 'dtype'):
            warnings.warn("ConfigNamespace missing 'dtype' attribute. Using default torch.float32.")
            dtype = torch.float32
        else:
            dtype = getattr(cfg, 'dtype')
        
        self.fc1 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False) # gate_proj
        self.fc2 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False) # up_proj
        self.fc3 = nn.Linear(hidden_dim, emb_dim, dtype=dtype, bias=False) # down_proj

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
        
        # Check for essential attributes with appropriate defaults and warnings
        if not hasattr(cfg, 'emb_dim'):
            warnings.warn("ConfigNamespace missing 'emb_dim' attribute. Using default value 4096.")
            emb_dim = 4096
        else:
            emb_dim = getattr(cfg, 'emb_dim')
            
        if not hasattr(cfg, 'dtype'):
            warnings.warn("ConfigNamespace missing 'dtype' attribute. Using default torch.float32.")
            dtype = torch.float32
        else:
            dtype = getattr(cfg, 'dtype')
            
        # Use RMSNorm from scratch code implementation
        rms_norm_eps = cfg.get("rms_norm_eps", 1e-5)
        
        self.norm1 = RMSNorm(emb_dim, eps=rms_norm_eps, dtype=dtype)
        self.norm2 = RMSNorm(emb_dim, eps=rms_norm_eps, dtype=dtype)

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
        self._initialized = False  # <-- Add this flag for checkpoint compatibility
        
        # Use .get() method or direct attribute access instead of dictionary-style access
        # Check for essential attributes and provide meaningful defaults with warnings
        if not hasattr(cfg, 'dim'):
            warnings.warn("ConfigNamespace missing 'dim' attribute. Using default value 4096.")
            self.emb_dim = 4096
        else:
            self.emb_dim = getattr(cfg, 'dim')
        
        # Handle dtype properly
        if not hasattr(cfg, 'dtype'):
            warnings.warn("ConfigNamespace missing 'dtype' attribute. Using default torch.float32.")
            self.dtype = torch.float32
        else:
            dtype_val = getattr(cfg, 'dtype')
            self.dtype = getattr(torch, dtype_val) if isinstance(dtype_val, str) else dtype_val
        
        # Access other attributes through getattr with defaults
        if not hasattr(cfg, 'n_heads'):
            warnings.warn("ConfigNamespace missing 'n_heads' attribute. Using default value 32.")
            self.n_heads = 32
        else:
            self.n_heads = getattr(cfg, 'n_heads')
            
        # Calculate head_dim directly rather than warning
        if hasattr(cfg, 'head_dim'):
            self.head_dim = getattr(cfg, 'head_dim')
        else:
            self.head_dim = self.emb_dim // self.n_heads
            print(f"Automatically set head_dim to {self.head_dim} (emb_dim/n_heads)")
        
        # Other parameters with defaults through .get()
        self.n_memory_slots = cfg.get('hopfield_memory_slots', 256)  # Default 256
        self.update_strategy = cfg.get('hopfield_update_strategy', 'replace_lru')  # Default: replace_lru
        self.num_updates = cfg.get('hopfield_num_updates', 1)  # How many memory slots to update at once
        self.combine_method = cfg.get('hopfield_combine_method', 'add')  # How to combine memory with output: add, gated_add, concat
        self.init_method = cfg.get('hopfield_init_method', 'zeros')  # zeros, embedding_sampling
        self.memory_update_freq = cfg.get('hopfield_memory_update_freq', 1)  # Default: update every doc
        self.update_counter = 0  # Internal counter for update frequency
        self.clamp_beta = cfg.get('hopfield_clamp_beta', True)  # Whether to clamp beta to [0, 1]
        
        # For beta scaling in attention calculation
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=self.dtype))  # Scalar parameter for attention temperature
        
        # FIXED: First project input to head dimension before applying further projections
        self.input_proj = nn.Linear(self.emb_dim, self.head_dim * self.n_heads, bias=False, dtype=self.dtype)
        
        # Projection layers for each head
        # These create query/key/value projections for memory attention
        self.q_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        # Layer norms for query and retrieved memory - FIXED: now using correct dimensions
        self.norm_query = RMSNorm(self.emb_dim, dtype=self.dtype)  # Apply to original input
        self.norm_retrieved = RMSNorm(self.head_dim * self.n_heads, dtype=self.dtype)  # For after reshaping
        
        # Memory combination mechanism
        if self.combine_method == 'concat':
            # When concatenating, need to project back to head_dim
            self.combine_proj = nn.Linear(self.head_dim * 2, self.head_dim, bias=False)
        elif self.combine_method == 'gated_add':
            # Gated addition with learned weighting
            self.combine_gate_proj = nn.Linear(self.head_dim * 2, self.head_dim, bias=False)
        
        # Memory slots: [n_heads, n_memory_slots, head_dim]
        # Initialize with zeros or later with embedding samples
        # Use a special deepspeed-compatible register_parameter approach
        stored_patterns = torch.zeros(self.n_heads, self.n_memory_slots, self.head_dim, dtype=self.dtype)
        
        # Ensure parameter dictionary has _in_forward for DeepSpeed compatibility
        self._orig_register_parameter = nn.Module.register_parameter
        def ds_register_parameter(module, name, param):
            module._orig_register_parameter(name, param)
            # After registering parameter, check if _parameters has _in_forward attribute
            if not hasattr(module._parameters, "_in_forward"):
                class ParametersDict(dict):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self._in_forward = False
                
                new_params = ParametersDict()
                for p_name, p in module._parameters.items():
                    new_params[p_name] = p
                module._parameters = new_params
        
        # Create the parameter using standard register_parameter
        nn.Module.register_parameter(self, 'storedpatterns', nn.Parameter(stored_patterns))
        
        # Then convert the parameters dict to a DeepSpeed-compatible one
        if not hasattr(self._parameters, "_in_forward"):
            class ParametersDict(dict):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._in_forward = False
            
            new_params = ParametersDict()
            for p_name, p in self._parameters.items():
                new_params[p_name] = p
            self._parameters = new_params
        
        # Access counts for LRU tracking - not a parameter, just a buffer
        self.register_buffer('access_counts', torch.zeros(self.n_heads, self.n_memory_slots, dtype=torch.long))
        
        # Register buffers for last state
        self.register_buffer('last_query_state', torch.zeros(1, self.n_heads, 1, self.head_dim, dtype=self.dtype))
        self.register_buffer('last_key_state', torch.zeros(1, self.n_heads, self.n_memory_slots, self.head_dim, dtype=self.dtype))
        self.register_buffer('last_value_state', torch.zeros(1, self.n_heads, self.n_memory_slots, self.head_dim, dtype=self.dtype))
        self.register_buffer('last_attn_probs', torch.zeros(1, self.n_heads, 1, self.n_memory_slots, dtype=self.dtype))
        self.register_buffer('last_attn_output', torch.zeros(1, self.n_heads, 1, self.head_dim, dtype=self.dtype))
        
        # Initialize storage for the last processed query for memory updates
        self._last_query_input = None
        self._last_retrieved_memory_norm = None
        
        # For gated update
        if self.update_strategy == "gated":
            self.gate_input_pooling = cfg.get('hopfield_gate_input_pooling', 'mean')
            gate_input_dim = self.emb_dim * 2  # concat of query and retrieved
            # Fix: output dimension should be n_heads * n_memory_slots, not n_heads * n_memory_slots * head_dim
            # This ensures gate_values_flat can be reshaped to [n_heads, n_memory_slots, 1] and then expanded
            gate_output_dim = self.n_heads * self.n_memory_slots
            self.gate_linear = nn.Linear(gate_input_dim, gate_output_dim, bias=False, dtype=self.dtype)
        
        # For update target method
        self.update_target_method = cfg.get('hopfield_update_target_method', 'avg_query')
        self.memory_update_lr = cfg.get('hopfield_memory_update_lr', 0.01)
        self.clamp_patterns = cfg.get('hopfield_clamp_patterns', None)

        # Initialize memory if requested method isn't 'zeros'
        if self.init_method != 'zeros':
            self.initialize_memory()
        
        self._initialized = True  # Mark as initialized

    def initialize_memory(self, embedding_matrix: Optional[torch.Tensor] = None):
        # Skip if already initialized
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
                 # Use different message based on scenario
                 if embedding_matrix is None and self.init_method == "embedding_sampling":
                     print("Embedding matrix not provided for 'embedding_sampling'. Using random initialization.")
                 else:
                     print(f"Using random initialization for memory patterns (method: {self.init_method}).")
                 # Use normal distribution for initialization
                 self.storedpatterns.data.normal_(mean=0.0, std=0.02)
        
        self._initialized = True


    def forward(self, query_input): # Removed mask pass-through for simplicity
        # query_input - [B, T, D]
        # Create empty last state if not initialized yet (first call during checkpointing)
        if not self._initialized and not self.training:
            self._clear_last_state()
            self._initialized = True
            
        batch_size, seq_len, _ = query_input.shape
        
        # Store query_input for later memory updates if in training mode
        if self.training:
            self._last_query_input = query_input.detach().clone()
        
        # Apply RMSNorm to the original input first
        normed_query = self.norm_query(query_input)
        
        # Project normalized input to head dimensions - NEW STEP
        projected_query = self.input_proj(normed_query)  # [B, T, H*D_h]
        
        # Reshape to prepare for multi-head processing
        reshaped_query = projected_query.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Now proceed with head-specific projections
        q = self.q_proj(reshaped_query)  # [B, T, H, D_h]
        
        # Transpose for attention calculation: [B, H, T, D_h]
        q = q.transpose(1, 2)
        
        # Project memory patterns for keys and values
        # [H, M, D_h] -> [H, M, D_h]
        k = self.k_proj(self.storedpatterns)
        v = self.v_proj(self.storedpatterns)
        
        # Scale query by beta parameter
        # Get beta and ensure proper shape [1, H, 1, 1]
        beta = self.beta
        if self.clamp_beta:
            # Option to clamp beta to reasonable range
            beta = torch.sigmoid(beta) # ensure 0-1 range
        
        # Apply scaled attention
        # q: [B, H, T, D_h], k: [H, M, D_h], v: [H, M, D_h]
        # Reshape k and v to [1, H, M, D_h] for batch operations
        k = k.unsqueeze(0)  # [1, H, M, D_h]
        v = v.unsqueeze(0)  # [1, H, M, D_h]
        
        # Scale query by beta
        q_scaled = q * beta
        
        # Calculate attention scores
        # q_scaled: [B, H, T, D_h], k: [1, H, M, D_h]
        # -> attn_scores: [B, H, T, M]
        attn_scores = torch.matmul(q_scaled, k.transpose(-1, -2))
        
        # Get attention probabilities with softmax along memory dimension
        attn_probs = torch.softmax(attn_scores, dim=-1)  # [B, H, T, M]
        
        # Calculate weighted sum of memory values using attention
        # attn_probs: [B, H, T, M], v: [1, H, M, D_h]
        # -> attn_output: [B, H, T, D_h]
        attn_output = torch.matmul(attn_probs, v)  # [B, H, T, D_h]
        
        # Reshape attention output: [B, H, T, D_h] -> [B, T, H, D_h] -> [B, T, H*D_h]
        # OR Keep multihead format for return
        attn_output = attn_output.transpose(1, 2)  # [B, T, H, D_h]
        
        # Remember current state for memory updates
        # Only store if we're in training mode to support update_memory later
        if self.training:
            # Make sure all last states have non-zero sizes
            # This is critical for gradient checkpointing to work properly
            self.last_query_state = q[:, :, -1:, :].detach().clone()  # Last token only
            self.last_key_state = k.detach().clone()
            self.last_value_state = v.detach().clone()
            self.last_attn_probs = attn_probs[:, :, -1:, :].detach().clone()  # Last token only
            self.last_attn_output = attn_output[:, -1:, :, :].detach().clone()  # Last token only
        
        # Reshape to [B, T, H*D_h] for normalization and combination
        attn_output_flat = attn_output.reshape(batch_size, seq_len, -1)
        
        # Apply normalization to retrieved memory
        retrieved_memory_norm = self.norm_retrieved(attn_output_flat)
        
        # Store retrieved memory for update calculations if in training
        if self.training:
            self._last_retrieved_memory_norm = retrieved_memory_norm.detach().clone()
        
        # Combine the retrieved memory with the original query
        memory_output = self._combine_memory_with_output(retrieved_memory_norm, query_input)
        
        # Update access counts for memory slots based on attention probabilities
        # Sum attention over batch and sequence dimensions
        # Using clone and detach to avoid affecting gradients
        if self.training:
            with torch.no_grad():
                # Sum over batch and sequence
                slot_access = attn_probs.detach().sum(dim=(0, 2))  # [H, M]
                self.access_counts += slot_access.long()
        
        return memory_output

    def update_memory(self):
        # Skip update if strategy is none or no input was processed
        if self.update_strategy in ["none", None] or \
           self._last_query_input is None or \
           self._last_query_input.numel() == 0: # Check if tensor is empty
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
            elif self.update_target_method == "max_query":
                if self._last_query_input is not None:
                    try:
                        # Get the maximum activation per feature across the batch and sequence length
                        # This captures the strongest features from the input
                        max_query, _ = torch.max(self._last_query_input.to(device, dtype), dim=0)
                        max_query, _ = torch.max(max_query.unsqueeze(0), dim=1)
                        
                        # Reshape to match the per-head structure
                        target_per_head = max_query.view(self.n_heads, self.head_dim)
                        # --- DEBUG PRINT --- 
                        print(f"[DEBUG update_memory max_query] storedpatterns shape: {self.storedpatterns.shape}", flush=True)
                        # --- END DEBUG --- 
                        update_target = target_per_head.unsqueeze(1).expand_as(self.storedpatterns)
                        print(f"Using max_query update target with shape {update_target.shape}")
                    except RuntimeError as e:
                        print(f"Error in max_query update: {e}")
                        # Fallback to avg_query if max_query fails
                        avg_query = torch.mean(self._last_query_input.to(device, dtype), dim=(0, 1))
                        target_per_head = avg_query.view(self.n_heads, self.head_dim)
                        # --- DEBUG PRINT --- 
                        print(f"[DEBUG update_memory max_query fallback] storedpatterns shape: {self.storedpatterns.shape}", flush=True)
                        # --- END DEBUG --- 
                        update_target = target_per_head.unsqueeze(1).expand_as(self.storedpatterns)
                        print("Falling back to avg_query method")
                else:
                    print("Warning: Missing query input for max_query update target.")

            elif self.update_target_method == "avg_retrieved":
                 if self._last_retrieved_memory_norm is not None:
                      avg_retrieved = torch.mean(self._last_retrieved_memory_norm.to(device, dtype), dim=(0, 1))
                      try:
                           target_per_head = avg_retrieved.view(self.n_heads, self.head_dim)
                           # --- DEBUG PRINT --- 
                           print(f"[DEBUG update_memory avg_retrieved] storedpatterns shape: {self.storedpatterns.shape}", flush=True)
                           # --- END DEBUG --- 
                           update_target = target_per_head.unsqueeze(1).expand_as(self.storedpatterns)
                      except RuntimeError as e: print(f"Error reshaping avg_retrieved: {e}")
                 else: print("Warning: Missing retrieved memory for 'avg_retrieved' target.")

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
                    elif self.gate_input_pooling == "attention":
                        # Implement attention-based pooling
                        try:
                            # Create a simple attention mechanism for pooling
                            # 1. Project the query input to get attention scores
                            if not hasattr(self, 'gate_attn_proj'):
                                self.gate_attn_proj = nn.Linear(self.emb_dim, 1, bias=False).to(device)
                            
                            # Shape: [B, T, D] -> [B, T, 1]
                            query_input_dev = self._last_query_input.to(device, dtype)
                            query_attn_scores = self.gate_attn_proj(query_input_dev)
                            
                            # Apply softmax over sequence dimension with temperature for stability
                            # Add small epsilon to avoid instability
                            temp = 1.0
                            query_attn_weights = torch.softmax(query_attn_scores / temp, dim=1)
                            
                            # Weighted sum to get pooled representation
                            # [B, T, 1] * [B, T, D] -> [B, D] after sum
                            query_pooled = torch.sum(query_attn_weights * query_input_dev, dim=1)
                            
                            # Ensure we have a valid representation by handling batch dimension properly
                            if query_pooled.dim() == 2 and query_pooled.size(0) > 0:
                                # Average over batch
                                query_pooled = torch.mean(query_pooled, dim=0)
                            elif query_pooled.dim() == 1:
                                # Already a single vector, no need to reduce
                                pass
                            else:
                                raise ValueError(f"Unexpected query_pooled shape: {query_pooled.shape}")
                            
                            # Similar for retrieved memory
                            if not hasattr(self, 'gate_retrieved_attn_proj'):
                                # Use the correct dimension for the projection
                                retrieved_dim = self._last_retrieved_memory_norm.size(-1)
                                self.gate_retrieved_attn_proj = nn.Linear(retrieved_dim, 1, bias=False).to(device)
                            
                            retrieved_mem_dev = self._last_retrieved_memory_norm.to(device, dtype)
                            retrieved_attn_scores = self.gate_retrieved_attn_proj(retrieved_mem_dev)
                            retrieved_attn_weights = torch.softmax(retrieved_attn_scores / temp, dim=1)
                            retrieved_pooled = torch.sum(retrieved_attn_weights * retrieved_mem_dev, dim=1)
                            
                            # Handle batch dimension for retrieved pooled representation
                            if retrieved_pooled.dim() == 2 and retrieved_pooled.size(0) > 0:
                                retrieved_pooled = torch.mean(retrieved_pooled, dim=0)
                            elif retrieved_pooled.dim() == 1:
                                # Already a single vector
                                pass
                            else:
                                raise ValueError(f"Unexpected retrieved_pooled shape: {retrieved_pooled.shape}")
                            
                            print(f"Using attention-based pooling for gate calculation (shapes: query={query_pooled.shape}, retrieved={retrieved_pooled.shape})")
                        except Exception as e:
                            print(f"Error in attention pooling: {e}, falling back to mean pooling")
                            query_pooled = torch.mean(self._last_query_input.to(device, dtype), dim=(0, 1))
                            retrieved_pooled = torch.mean(self._last_retrieved_memory_norm.to(device, dtype), dim=(0, 1))

                    if query_pooled is not None and retrieved_pooled is not None:
                         gate_input = torch.cat((query_pooled, retrieved_pooled), dim=-1).to(self.gate_linear.weight.dtype)
                         # Print the actual needed output size
                         num_elements = self.n_heads * self.n_memory_slots * self.head_dim
                         
                         # Check if gate_linear has the right output size
                         required_output_size = self.n_heads * self.n_memory_slots
                         current_output_size = self.gate_linear.out_features
                         
                         # Recreate gate_linear if output shape is wrong
                         if current_output_size != required_output_size:
                             print(f"Recreating gate_linear: current size {current_output_size}, required size {required_output_size}")
                             self.gate_linear = nn.Linear(gate_input.size(-1), required_output_size, bias=False, dtype=self.dtype).to(device)
                         
                         # Get gate values
                         gate_values_flat = torch.sigmoid(self.gate_linear(gate_input))
                         print(f"Gate values shape: {gate_values_flat.shape}, stored patterns: {self.storedpatterns.shape}")
                         
                         # Reshape gate to match stored patterns - this time with the correct shape
                         gate = gate_values_flat.view(self.n_heads, self.n_memory_slots, 1).expand_as(self.storedpatterns).to(dtype)
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
        """Initialize/clear the last state tensors for gradient checkpointing safety"""
        batch_size = 1  # Dummy batch size for empty tensors
        device = self.storedpatterns.device
        dtype = self.storedpatterns.dtype
        
        # Create empty tensors with the right shape but zero size in time dimension
        # This ensures tensor metadata consistency during checkpointing
        self.last_query_state = torch.zeros(batch_size, self.n_heads, 0, self.head_dim, 
                                           device=device, dtype=dtype)
        self.last_key_state = torch.zeros(1, self.n_heads, self.n_memory_slots, self.head_dim,
                                         device=device, dtype=dtype)
        self.last_value_state = torch.zeros(1, self.n_heads, self.n_memory_slots, self.head_dim,
                                           device=device, dtype=dtype)
        self.last_attn_probs = torch.zeros(batch_size, self.n_heads, 0, self.n_memory_slots,
                                          device=device, dtype=dtype)
        self.last_attn_output = torch.zeros(batch_size, 0, self.n_heads, self.head_dim,
                                           device=device, dtype=dtype)

    def reset_memory(self):
        """Clears the internal state used for memory updates."""
        self._clear_last_state()
        # Optional: Reset any other state if needed (e.g., accumulated statistics)
        print("Hopfield memory state cleared.")

    def _combine_memory_with_output(self, retrieved_memory, original_input):
        """Combine retrieved memory with original input according to config"""
        # retrieved_memory is [B, T, H*D_h] - already flattened and normalized
        # original_input is [B, T, D]
        
        # Combine based on method
        if self.combine_method == "add":
            # Simple addition - need to project retrieved memory to match original input dimension
            if retrieved_memory.size(-1) != original_input.size(-1):
                # Add a projection layer if needed
                if not hasattr(self, 'output_proj'):
                    self.output_proj = nn.Linear(self.head_dim * self.n_heads, self.emb_dim, 
                                               bias=False, dtype=self.dtype).to(retrieved_memory.device)
                retrieved_memory = self.output_proj(retrieved_memory)
            combined = original_input + retrieved_memory
            
        elif self.combine_method == "gated_add":
            # Gated addition using a sigmoid gate
            # Add projection layer if needed
            if retrieved_memory.size(-1) != original_input.size(-1):
                if not hasattr(self, 'output_proj'):
                    self.output_proj = nn.Linear(self.head_dim * self.n_heads, self.emb_dim, 
                                               bias=False, dtype=self.dtype).to(retrieved_memory.device)
                retrieved_memory = self.output_proj(retrieved_memory)
                
            # Need to adjust gate projection if dimensions don't match
            if not hasattr(self, 'combine_gate_proj') or self.combine_gate_proj.in_features != (original_input.size(-1) + retrieved_memory.size(-1)):
                self.combine_gate_proj = nn.Linear(original_input.size(-1) + retrieved_memory.size(-1), 
                                                original_input.size(-1), bias=False, 
                                                dtype=self.dtype).to(retrieved_memory.device)
                
            gate = torch.sigmoid(self.combine_gate_proj(torch.cat([original_input, retrieved_memory], dim=-1)))
            combined = original_input + gate * retrieved_memory
            
        elif self.combine_method == "concat_mlp":
            # Concatenate and project
            # Need to adjust combine_proj if dimensions don't match
            if not hasattr(self, 'combine_proj') or self.combine_proj.in_features != (original_input.size(-1) + retrieved_memory.size(-1)):
                self.combine_proj = nn.Linear(original_input.size(-1) + retrieved_memory.size(-1), 
                                           original_input.size(-1), bias=False, 
                                           dtype=self.dtype).to(retrieved_memory.device)
            
            concat = torch.cat([original_input, retrieved_memory], dim=-1)
            combined = self.combine_proj(concat)
        else:
            # Default to addition with projection if needed
            if retrieved_memory.size(-1) != original_input.size(-1):
                if not hasattr(self, 'output_proj'):
                    self.output_proj = nn.Linear(self.head_dim * self.n_heads, self.emb_dim, 
                                               bias=False, dtype=self.dtype).to(retrieved_memory.device)
                retrieved_memory = self.output_proj(retrieved_memory)
            combined = original_input + retrieved_memory
            
        return combined

# --- HAT Model (Using scratch structure) ---
class HopfieldLlama3Model(nn.Module, GenerationMixin):
    main_input_name = "input_ids"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg # Keep original dict if needed elsewhere
        cfg.setdefault('model_type', 'custom_llama')
        
        # FIXED: Check if cfg is a dict or an object with attributes
        is_dict = isinstance(cfg, dict)
        
        # FIXED: Ensure dimension settings are explicit
        if is_dict:
            # Dictionary access for regular dict
            if 'emb_dim' in cfg:
                cfg['dim'] = cfg['emb_dim']
                print(f"Setting cfg['dim'] to emb_dim: {cfg['dim']}")
                
            # Make sure hopfield_heads is properly handled
            if 'hopfield_heads' in cfg:
                cfg['n_heads'] = cfg['hopfield_heads'] 
                print(f"Setting n_heads from hopfield_heads: {cfg['hopfield_heads']}")
        else:
            # Attribute access for objects
            if not hasattr(cfg, 'dim') and hasattr(cfg, 'emb_dim'):
                cfg.dim = cfg.emb_dim
                print(f"Setting cfg.dim to emb_dim: {cfg.emb_dim}")
                
            # Make sure hopfield_heads is set if needed for HopfieldMemoryLayer
            if hasattr(cfg, 'hopfield_heads') and not hasattr(cfg, 'n_heads'):
                cfg.n_heads = cfg.hopfield_heads
                print(f"Setting cfg.n_heads from hopfield_heads: {cfg.hopfield_heads}")
        
        # Create config namespace from dict if needed
        if is_dict:
            self.config = ConfigNamespace(**cfg)
        else:
            self.config = cfg  # Already an object with attributes
            
        self.config.is_encoder_decoder = False # Explicitly set for GenerationMixin compatibility

        # --- REVISED APPROACH ---
        # Set use_cache on the config object (standard)
        self.config.use_cache = False
        # Set _supports_cache_class directly on the model instance
        self._supports_cache_class = False # <<< Set directly on self
        # --- END REVISION ---

        # Ensure we have emb_dim set in the config
        if not hasattr(self.config, 'emb_dim') and hasattr(self.config, 'dim'):
            self.config.emb_dim = self.config.dim
            print(f"Setting config.emb_dim from dim: {self.config.dim}")

        # Override register_parameter for DeepSpeed compatibility
        self._orig_register_parameter = self.register_parameter
        def deepspeed_compatible_register_parameter(name, param):
            self._orig_register_parameter(name, param)
            # Ensure _parameters has the _in_forward attribute for DeepSpeed
            if not hasattr(self._parameters, "_in_forward"):
                try:
                    class ParametersDict(dict):
                        def __init__(self, *args, **kwargs):
                            super().__init__(*args, **kwargs)
                            self._in_forward = False
                    
                    new_params = ParametersDict()
                    for p_name, p in self._parameters.items():
                        new_params[p_name] = p
                    self._parameters = new_params
                except Exception as e:
                    print(f"Error enhancing parameters dict: {e}")
        
        # Instead of replacing register_parameter, just make sure our parameters dict has _in_forward
        # Check the parameters dict after initialization
        if not hasattr(self._parameters, "_in_forward"):
            try:
                class ParametersDict(dict):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self._in_forward = False
                
                new_params = ParametersDict()
                for p_name, p in self._parameters.items():
                    new_params[p_name] = p
                self._parameters = new_params
                print("Applied DeepSpeed compatibility to HopfieldLlama3Model parameters")
            except Exception as e:
                print(f"Error enhancing parameters dict: {e}")

        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.emb_dim, dtype=self.config.dtype)
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.config.n_layers)]
        )
        
        # FIXED: Ensure Hopfield memory gets created with correct dimensions
        # Make hopfield_heads available if requested in config
        if hasattr(self.config, 'hopfield_heads'):
            hopfield_config = deepcopy(self.config)
            hopfield_config.n_heads = self.config.hopfield_heads
            print(f"Using {hopfield_config.n_heads} heads for Hopfield memory from hopfield_heads")
        else:
            hopfield_config = self.config
            
        # Create Hopfield memory if configured
        if getattr(self.config, 'use_hopfield_memory', False):
            self.hopfield_memory = HopfieldMemoryLayer(hopfield_config)
            # Special handling for memory initialization if embedding access is needed
            if hasattr(self.config, 'initialize_memory_from_embedding') and self.config.initialize_memory_from_embedding:
                print("Will initialize Hopfield memory from embedding matrix once it's accessible...")
                self._initialize_memory_pending = True
            else:
                print("Initializing Hopfield memory with random patterns")
                self._initialize_memory_pending = False
                # Initialize memory with None since we don't have embedding access yet
                self.hopfield_memory.initialize_memory(None)  
        else:
            self.hopfield_memory = None
            self._initialize_memory_pending = False
            print("No Hopfield memory layer created (use_hopfield_memory=False)")
        
        self.output_norm = RMSNorm(self.config.emb_dim, dtype=self.config.dtype)
        self.lm_head = nn.Linear(self.config.emb_dim, self.config.vocab_size, bias=False, dtype=self.config.dtype)
        
        # For hopfield_memory, initialize access counts with zeros
        if self.hopfield_memory is not None:
            n_memory_slots = getattr(self.config, 'hopfield_memory_slots', 1024)
            n_heads = getattr(self.config, 'hopfield_heads', self.config.n_heads)
            self.hopfield_memory.access_counts = torch.zeros(n_heads, n_memory_slots, dtype=torch.long)
            print(f"Hopfield memory initialized with {n_memory_slots} slots across {n_heads} heads")
        
        # Tie weights of token embedding and final projection
        # Do this at the end to ensure embedding has been initialized
        self.resize_token_embeddings(self.config.vocab_size) # This method will handle tying

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
        old_output = self.lm_head
        self.lm_head = nn.Linear(self.config.emb_dim, new_num_tokens, bias=False, dtype=self.config.dtype)
        
        # Copy weights for output head
        if old_output is not None and old_num_tokens > 0:
            self.lm_head.weight.data[:old_num_tokens] = old_output.weight.data
            
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

        h_final = self.output_norm(h_hopfield)
        logits = self.lm_head(h_final)

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
        """Initialize the Hopfield memory layer, potentially using embedding matrix"""
        if self.hopfield_memory is not None and self._initialize_memory_pending:
            print("Initializing Hopfield memory from embedding matrix...")
            try:
                # Get the embedding matrix from tok_emb
                embedding_matrix = self.tok_emb.weight.detach().clone()
                print(f"Obtained embedding matrix with shape {embedding_matrix.shape}")
                
                # Initialize memory with embedding matrix
                self.hopfield_memory.initialize_memory(embedding_matrix)
                self._initialize_memory_pending = False
                print(f"Hopfield memory initialized with {self.hopfield_memory.n_memory_slots} slots")
            except Exception as e:
                print(f"Error initializing Hopfield memory from embedding matrix: {e}")
                print("Falling back to random initialization")
                self.hopfield_memory.initialize_memory(None)
                self._initialize_memory_pending = False
        elif self.hopfield_memory is not None:
            print("Hopfield memory already initialized")
        else:
            print("No Hopfield memory layer exists")

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
        lm_head_present = isinstance(getattr(self, "lm_head", None), nn.Linear)
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

    # Create a copy of the config to avoid modifying the original
    config = config.copy()

    # --- Ensure critical numeric params from config dict are floats ---
    for key in ['rms_norm_eps', 'rope_base', 'learning_rate', 'hopfield_lr_multiplier', 
                'adam_epsilon', 'weight_decay', 'warmup_ratio']:
        if key in config:
            try:
                config[key] = float(config[key])
            except (ValueError, TypeError):
                # Default values if conversion fails
                defaults = {
                    'rms_norm_eps': 1e-5,
                    'rope_base': 500000.0,
                    'learning_rate': 3e-5,
                    'hopfield_lr_multiplier': 1.0,
                    'adam_epsilon': 1e-8,
                    'weight_decay': 0.0,
                    'warmup_ratio': 0.1
                }
                print(f"Warning: Could not convert {key} to float. Using default: {defaults.get(key)}")
                config[key] = defaults.get(key, 0.0)
    
    # Ensure dimensions are set correctly
    if 'emb_dim' in config and 'dim' not in config:
        config['dim'] = config['emb_dim']
        print(f"Setting config['dim'] to emb_dim: {config['dim']}")
    
    if 'hopfield_heads' in config and 'n_heads' not in config:
        config['n_heads'] = config['hopfield_heads']
        print(f"Setting n_heads from hopfield_heads: {config['hopfield_heads']}")

    # Determine torch dtype
    dtype_str = config.get("model_dtype", "float32")
    if dtype_str == "bfloat16" and torch.cuda.is_bf16_supported(): dtype = torch.bfloat16
    elif dtype_str == "float16": dtype = torch.float16
    else: dtype = torch.float32
    config["dtype"] = dtype

    print(f"Initializing HopfieldLlama3Model (Scratch Base) with dtype: {dtype}")
    # Initialize OUR model structure
    try:
        model = HopfieldLlama3Model(config)
        print("Model structure created successfully")
    except Exception as e:
        print(f"Error creating model structure: {e}")
        print(f"Config keys available: {list(config.keys())}")
        # Try to provide more details about the structure
        if 'emb_dim' in config:
            print(f"emb_dim: {config['emb_dim']}")
        if 'n_heads' in config:
            print(f"n_heads: {config['n_heads']}")
        if 'hopfield_heads' in config:
            print(f"hopfield_heads: {config['hopfield_heads']}")
        if 'n_layers' in config:
            print(f"n_layers: {config['n_layers']}")
        raise

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
                 # Don't use resume_download parameter due to deprecation warning
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
        elif key == "model.norm.weight": new_key = "output_norm.weight"
        elif key == "lm_head.weight": new_key = "lm_head.weight"

        adapted_state_dict[new_key] = value.to(dtype) # Convert dtype during adaptation

    # Load the state dict
    load_result = model.load_state_dict(adapted_state_dict, strict=False)
    
    # Fix for "missing non-Hopfield keys" warning
    # Only report non-hopfield missing keys and don't report lm_head.weight if we'll tie weights
    missing = [k for k in load_result.missing_keys if not k.startswith("hopfield_memory.") and k != "lm_head.weight"]
    unexpected = load_result.unexpected_keys
    if missing: warnings.warn(f"Weight loading missing keys: {missing}")
    if unexpected: warnings.warn(f"Weight loading unexpected keys: {unexpected}")
    print("Base model weights loaded into HAT structure.")

    # Tie weights if necessary (standard for Llama)
    # Explicitly tie weights even if lm_head.weight was missing in the state_dict
    print("Applying weight tying for output head.")
    model.lm_head.weight = model.tok_emb.weight

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