import einx
import math
import torch
from torch import nn

def get_truncated_normal(first: int, second: int):
    std = 2 / (first + second)
    return nn.Parameter(
        nn.init.trunc_normal_(
            torch.empty(first, second), 
            mean=0, std=std, a=-3*std, b=3*std)
    )

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, weight: torch.Tensor | None = None,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        # notice: out_features first since that's how the test cases do it
        if weight is None:
            self.weight = get_truncated_normal(out_features, in_features)
        else:
            self.weight = nn.Parameter(weight)

    def assign_weights(self, weight: torch.Tensor):
        with torch.no_grad():
            self.weight.copy_(weight)
    
    def forward(self, x: torch.Tensor):
        return einx.dot("... a, b a -> ... b", x, self.weight)

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = get_truncated_normal(num_embeddings, embedding_dim)
    
    def assign_weights(self, weight: torch.Tensor):
        with torch.no_grad():
            self.weight.copy_(weight)
    
    def forward(self, token_ids: torch.Tensor):
        # every element of token_ids (integer tensor) is viewed as an index into [a] axis
        # and then we are getting a b vector for each one
        return einx.get_at("[a] b, ... -> ... b", self.weight, token_ids)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def assign_weights(self, weight: torch.Tensor):
        with torch.no_grad():
            self.weight.copy_(weight)
    
    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        squared_mean = einx.elementwise("...", x, op=lambda x: x**2).mean(dim=-1, keepdim=True)
        result = einx.multiply("... [a]", x * (squared_mean + self.eps).rsqrt(), self.weight)
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: None | int = None, 
                 device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        if not d_ff:
            self.d_ff = 8 * d_model // 3
        else:
            self.d_ff = d_ff
        self.w1 = Linear(in_features=d_model, out_features=d_ff)
        self.w2 = Linear(in_features=d_ff, out_features=d_model)
        self.w3 = Linear(in_features=d_model, out_features=d_ff)
        
    def assign_weights(self, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor):
        with torch.no_grad():
            self.w1.assign_weights(w1)
            self.w2.assign_weights(w2)
            self.w3.assign_weights(w3)
        
    def activation(self, x: torch.Tensor):
        return x * x.sigmoid()
    
    def forward(self, x: torch.Tensor):
        return self.w2(self.activation(self.w1(x)) * self.w3(x))


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        self.d_k = d_k
        self.device = device

        # create buffer of pre-computed sines and cosines
        i_range = torch.arange(start=0, end=max_seq_len, step=1, dtype=torch.float)
        thetas = 1 / (theta**(2*(torch.arange(start=0, end=d_k//2, step=1))/d_k))
        theta_i_k = einx.elementwise("a, b -> a b", i_range, thetas, op=lambda a, b: a * b)
        cosines = torch.cos(theta_i_k).repeat_interleave(2, dim=-1).to(device)
        sines = torch.sin(theta_i_k).repeat_interleave(2, dim=-1).to(device)
        self.register_buffer("cosines", cosines, persistent=False)
        self.register_buffer("sines", sines, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # we are assuming this as the input from the tests are wack
        assert len(token_positions.shape) == 1

        selected_cos = self.cosines.index_select(0, token_positions)
        selected_sin = self.sines.index_select(0, token_positions)

        x_r = einx.rearrange("... (e f) -> ... e f", x, f=2)
        x_r = torch.stack([-x_r[..., 1], x_r[..., 0]], dim=-1)
        x_r = einx.rearrange("... e f -> ... (e f)", x_r, f=2)

        return selected_cos * x + selected_sin * x_r

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                 mask: torch.Tensor = None):
    pre_softmax = einx.dot("... q [d_k], ... k [d_k] -> ... q k", Q, K) / math.sqrt(Q.shape[-1])
    if mask is None:
        mask = torch.full(pre_softmax.shape, True)
    # NEED TO MAKE THIS DIFFERENTIABLE
    masked = einx.where("... d e, ... d e, ", mask, pre_softmax, -float("Inf"))
    post_softmax = softmax(masked, dim=-1)
    return einx.dot("... q [k], ... [k] d_v -> ... q d_v", post_softmax, V)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: str=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.device = device

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
    
    def assign_weights(self, q_proj: torch.Tensor, k_proj: torch.Tensor, v_proj: torch.Tensor,
                       output_proj: torch.Tensor):
            self.q_proj = Linear(q_proj.shape[0], q_proj.shape[1], weight=q_proj)
            self.k_proj = Linear(k_proj.shape[0], k_proj.shape[1], weight=k_proj)
            self.v_proj = Linear(v_proj.shape[0], v_proj.shape[1], weight=v_proj)
            self.output_proj = Linear(output_proj.shape[0], output_proj.shape[1], weight=output_proj)

    def set_rope(self, theta: float, max_seq_length: int):
        self.r = RotaryPositionEmbedding(theta, self.d_k, max_seq_length, self.device)
    
    def forward(self, x: torch.Tensor, rope: bool = False, token_positions : torch.Tensor = None):
        sequence_length = x.shape[-2]
        combined = einx.rearrange("q d_in, k d_in, v d_in -> (q + k + v) d_in", 
                                  self.q_proj.weight, self.k_proj.weight, self.v_proj.weight)
        multiplied = einx.dot("ds [in], ... s [in] -> ... s ds", combined, x)
        q, k, v = einx.rearrange("... (d + d + d) -> ... d, ... d, ... d", multiplied)

        if rope:
            if token_positions is None:
                token_positions = torch.arange(sequence_length).to(self.device)
            assert self.r is not None

        mask = torch.tril(torch.ones(sequence_length, sequence_length)).to(device=self.device, dtype=torch.bool)
        if rope:
            res = einx.vmap(
                "... [s] (h [c]), ... [s] (h [c]), ... [s] (h [c]) -> ... [s] (h [c])", 
                q, k, v, h=self.num_heads,
                op=lambda q, k, v: scaled_dot_product_attention(
                    self.r(q, token_positions), self.r(k, token_positions), v, mask
                )
            )
        else:
            res = einx.vmap(
                "... [s] (h [c]), ... [s] (h [c]), ... [s] (h [c]) -> ... [s] (h [c])", 
                q, k, v, h=self.num_heads,
                op=lambda q, k, v: scaled_dot_product_attention(q, k, v, mask)
            )
        return einx.dot("d_m d_v, ... s d_v -> ... s d_m", self.output_proj.weight, res)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float,
                 device: str=None):
        super().__init__()

        self.attn = MultiHeadSelfAttention(d_model, num_heads, device)
        self.attn.set_rope(theta, max_seq_len)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        
    def assign_weights(self, weights: dict[str, torch.Tensor]):
        self.load_state_dict(weights)
    
    def forward(self, x: torch.Tensor):
        x = x + self.attn.forward(self.ln1(x), rope=True)
        return x + self.ffn(self.ln2(x))

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int,
                 num_heads: int, d_ff: int, rope_theta: float, device: str=None):
        super().__init__()

        self.context_length = context_length

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(
            d_model, num_heads, d_ff, context_length, rope_theta, device
        ) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def assign_weights(self, weights: dict[str, torch.Tensor]):
        self.load_state_dict(weights)

    def forward(self, x: torch.Tensor):
        assert x.shape[-1] <= self.context_length
        x = self.token_embeddings(x)
        for transformer_block in self.layers:
            x = transformer_block(x)
        return self.lm_head(self.ln_final(x))

def softmax(x: torch.Tensor, dim: int):
    normalized_exp = (x - x.max(dim=dim, keepdim=True).values).exp()
    den = normalized_exp.sum(dim=dim, keepdim=True)
    return normalized_exp / den

def log_probabilities(x: torch.Tensor, dim: int):
    maxes = x.max(dim=dim, keepdim=True).values
    return (x - maxes) - (x - maxes).exp().sum(dim=dim, keepdim=True).log()

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    return -log_probabilities(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).mean()