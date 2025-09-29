import torch
import torch.nn as nn
import math


def rotate_every_two(x):
    """
    Helper function that rotates every two dimensions.
    Splits the last dimension into pairs and performs a 90° rotation.
    """
    x1 = x[..., ::2]  # even indices
    x2 = x[..., 1::2]  # odd indices
    # For each pair, rotate: (x1, x2) -> (-x2, x1)
    x_rotated = torch.stack((-x2, x1), dim=-1)
    return x_rotated.flatten(-2)

def apply_rotary_pos_emb(x, sin, cos):
    """
    Applies rotary positional embeddings.
    
    Args:
      x: Tensor of shape [batch, seq_len, dim]
      sin: Sinusoidal tensor of shape [1, seq_len, dim]
      cos: Cosine tensor of shape [1, seq_len, dim]
    
    Returns:
      Tensor with rotary embeddings applied.
    """
    return (x * cos) + (rotate_every_two(x) * sin)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, dim, freq):
        """
        Initializes rotary positional embeddings.
        
        Args:
          dim: Dimensionality of the model (should be even).
          max_seq_len: Maximum sequence length expected.
        """
        super().__init__()
        # Create inverse frequency vector (shape: [dim/2])
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)) 
        # Positions (shape: [max_seq_len])
        positions = torch.arange(0, max_seq_len//freq).float()
        positions = positions.repeat_interleave(freq)
        # Outer product to get sinusoid arguments: [max_seq_len, dim/2]
        sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)
        # Precompute sin and cos values; unsqueeze to shape [1, max_seq_len, dim/2]
        sin = torch.sin(sinusoid_inp).unsqueeze(0)
        cos = torch.cos(sinusoid_inp).unsqueeze(0)
        # For convenience, we repeat each value to match the full dimension later.
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)

    def forward(self, x):
        """
        Applies rotary positional embeddings to the input tensor.
        
        Args:
          x: Tensor of shape [batch, seq_len, dim]
          
        Returns:
          Tensor of shape [batch, seq_len, dim] with RoPE applied.
        """
        batch_size, seq_len, dim = x.size()
        # Get precomputed sin/cos values for the current sequence length
        sin = self.sin[:, :seq_len, :]  # shape: [1, seq_len, dim/2]
        cos = self.cos[:, :seq_len, :]  # shape: [1, seq_len, dim/2]
        # Expand sin and cos to match the dimension of x by repeating each element along the last dim
        sin = sin.repeat_interleave(2, dim=-1)  # shape: [1, seq_len, dim]
        cos = cos.repeat_interleave(2, dim=-1)  # shape: [1, seq_len, dim]
        return apply_rotary_pos_emb(x, sin, cos)


class RotaryPositionalEmbedding2D(nn.Module):
    def __init__(self, height, width, dim):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even"
        assert dim % 2 == 0 and dim % 2 == 0, "Currently assuming dim is split equally into 2 axes"

        self.dim = dim
        half_dim = dim // 2

        inv_freq_h = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        inv_freq_w = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))

        pos_h = torch.arange(height, dtype=torch.float32)
        pos_w = torch.arange(width, dtype=torch.float32)

        sin_inp_h = torch.einsum("i,j->ij", pos_h, inv_freq_h)
        sin_inp_w = torch.einsum("i,j->ij", pos_w, inv_freq_w)

        sin_h = torch.sin(sin_inp_h)
        cos_h = torch.cos(sin_inp_h)
        sin_w = torch.sin(sin_inp_w)
        cos_w = torch.cos(sin_inp_w)

        self.register_buffer("sin_h", sin_h)
        self.register_buffer("cos_h", cos_h)
        self.register_buffer("sin_w", sin_w)
        self.register_buffer("cos_w", cos_w)

    def forward(self, x):
        """
        Args:
          x: [batch, H, W, dim]
        """
        B, H, W, D = x.shape
        D_half = D // 2
        x_h, x_w = x[..., :D_half], x[..., D_half:]

        # For height
        sin_h = self.sin_h[:H].unsqueeze(1).repeat(1, W, 1)  # [H, W, D_half//2]
        cos_h = self.cos_h[:H].unsqueeze(1).repeat(1, W, 1)
        sin_h = sin_h.unsqueeze(0).repeat(B, 1, 1, 1)
        cos_h = cos_h.unsqueeze(0).repeat(B, 1, 1, 1)

        x_h = apply_rotary_pos_emb(x_h, sin_h.repeat_interleave(2, dim=-1), cos_h.repeat_interleave(2, dim=-1))

        # For width
        sin_w = self.sin_w[:W].unsqueeze(0).repeat(H, 1, 1)  # [H, W, D_half//2]
        cos_w = self.cos_w[:W].unsqueeze(0).repeat(H, 1, 1)
        sin_w = sin_w.unsqueeze(0).repeat(B, 1, 1, 1)
        cos_w = cos_w.unsqueeze(0).repeat(B, 1, 1, 1)

        x_w = apply_rotary_pos_emb(x_w, sin_w.repeat_interleave(2, dim=-1), cos_w.repeat_interleave(2, dim=-1))

        return torch.cat([x_h, x_w], dim=-1)


class RotaryPositionalEmbedding3D(nn.Module):
    def __init__(self, time, height, width, dim, freq):
        super().__init__()
        self.d_each = dim 

        def create_freq(size, d_each, freq=1):
            inv_freq = 1.0 / (10000 ** (torch.arange(0, d_each, 2).float() / d_each))
            pos = torch.arange(size//freq).float()
            pos = pos.repeat_interleave(freq)
            sin_inp = torch.einsum("i,j->ij", pos, inv_freq)
            return torch.sin(sin_inp), torch.cos(sin_inp)

        sin_t, cos_t = create_freq(time, self.d_each, freq=freq)
        sin_h, cos_h = create_freq(height, self.d_each)
        sin_w, cos_w = create_freq(width, self.d_each)

        self.register_buffer("sin_t", sin_t)
        self.register_buffer("cos_t", cos_t)
        self.register_buffer("sin_h", sin_h)
        self.register_buffer("cos_h", cos_h)
        self.register_buffer("sin_w", sin_w)
        self.register_buffer("cos_w", cos_w)

    def forward(self, x):
        """
        x: [B, T, H, W, dim]
        """
        # import ipdb; ipdb.set_trace()
        B, T, H, W, D = x.shape

        x_t, x_h, x_w = x, x, x

        # Time axis
        sin_t = self.sin_t[:T].view(1, T, 1, 1, -1).repeat(B, 1, H, W, 1)
        cos_t = self.cos_t[:T].view(1, T, 1, 1, -1).repeat(B, 1, H, W, 1)
        x_t = apply_rotary_pos_emb(x_t, sin_t.repeat_interleave(2, dim=-1), cos_t.repeat_interleave(2, dim=-1))

        # Height axis
        sin_h = self.sin_h[:H].view(1, 1, H, 1, -1).repeat(B, T, 1, W, 1)
        cos_h = self.cos_h[:H].view(1, 1, H, 1, -1).repeat(B, T, 1, W, 1)
        x_h = apply_rotary_pos_emb(x_h, sin_h.repeat_interleave(2, dim=-1), cos_h.repeat_interleave(2, dim=-1))

        # Width axis
        sin_w = self.sin_w[:W].view(1, 1, 1, W, -1).repeat(B, T, H, 1, 1)
        cos_w = self.cos_w[:W].view(1, 1, 1, W, -1).repeat(B, T, H, 1, 1)
        x_w = apply_rotary_pos_emb(x_w, sin_w.repeat_interleave(2, dim=-1), cos_w.repeat_interleave(2, dim=-1))

        return x_h + x_t + x_w



class Qwen2_5OmniRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2_5OmniThinkerConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen2_5Omni has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    

class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs
    
def rot_pos_emb(self, grid_thw):
    pos_ids = []
    for t, h, w in grid_thw:
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid_size = grid_thw[:, 1:].max()
    rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
    rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
    return rotary_pos_emb
    

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

        
if __name__ == "__main__":
    pos = RotaryPositionalEmbedding3D(100, 64, 64, 64, 25)
    x = torch.ones((1,100,64,64,64))
    print(pos(x).size())
