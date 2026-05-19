"""
Transformer Encoder architecture. Up to date on Feb. 2026. 
The attention mechanism is still vanilla (besides gating).
Other than that it uses RMSNorm, Rope encodings, SwiGLU MLPs,
and gated attention.
The transformer can be configured for MLM, causal learning, or 
to return the mean embedding for contrastive learning.
Variable length sequences are padded to the length of the longes one.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # x: (..., dim)
        # Compute in float32 for numerical stability when float16 autocast:
        # eps=1e-8 underflows to 0 in float16 (min subnormal ~6e-8), causing
        # division by zero and NaN propagation through attention later.
        orig_dtype = x.dtype
        x_fp32 = x.float()
        denom = torch.sqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_fp32 * (self.scale.float() / denom)).to(orig_dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim_head: int, base: int = 10000):
        super().__init__()
        assert dim_head % 2 == 0, f"dim_head ({dim_head}) must be even for RoPE"
        self.dim_head = dim_head
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_head, 2).float() / dim_head))
        self.register_buffer("inv_freq", inv_freq)
        self._cached_seq_len = 0
        self.register_buffer("_cached_cos", None, persistent=False)
        self.register_buffer("_cached_sin", None, persistent=False)

    def get_embed(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        # Always cache to the length of the longest sequence seen so far.
        if seq_len <= self._cached_seq_len and self._cached_cos is not None:
            return self._cached_cos[:seq_len], self._cached_sin[:seq_len]
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device=device, dtype=dtype))
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cached_cos = emb.cos()
        self._cached_sin = emb.sin()
        self._cached_seq_len = seq_len
        return self._cached_cos, self._cached_sin

def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    def rotate(x: torch.Tensor):
        b, seq, nh, hd = x.shape
        x2 = x.reshape(b, seq, nh, hd // 2, 2)
        x_rot = torch.stack((-x2[..., 1], x2[..., 0]), dim=-1)
        return x_rot.reshape(b, seq, nh, hd)

    return (q * cos + rotate(q) * sin, k * cos + rotate(k) * sin)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout_p = float(dropout)
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                rotary: Optional[RotaryEmbedding] = None, is_causal:Optional[bool] = False):
        b, seq, _ = x.shape
        qkv = self.qkv(x).view(b, seq, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        if rotary is not None:
            cos, sin = rotary.get_embed(seq, x.device, x.dtype)
            q, k = apply_rope(q, k, cos, sin)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        #with torch.backends.cuda.enable_flash_sdp(enabled=True):
        out = F.scaled_dot_product_attention(q, k, v,
                                                attn_mask=attn_mask,
                                                dropout_p=self.dropout_p if self.training else 0.0,
                                                is_causal=is_causal)

        out = out.permute(0, 2, 1, 3).reshape(b, seq, self.dim)
        return self.out(out)

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x: torch.Tensor, valid_mask=None):
        a, b = self.w1(x).chunk(2, dim=-1)
        x = self.act(a) * b
        x = self.dropout(x)
        return self.w2(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadAttention(dim, n_heads, dropout=dropout)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, hidden_dim, dropout=dropout)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        # Gaiting as in "Gated Attention for Large Language Models" Qiu et al. Neurips 2025
        self.gate_linear = nn.Linear(dim, dim, bias=True)
        nn.init.xavier_uniform_(self.gate_linear.weight)


    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, rotary: Optional[RotaryEmbedding] = None,
                is_causal:Optional[bool]=False):
        x_ln = self.norm1(x)
        gate_scores = torch.sigmoid(self.gate_linear(x_ln))
        attn_out = self.attn(x_ln, attn_mask=attn_mask, rotary=rotary, is_causal=is_causal)
        attn_out = attn_out *gate_scores
        x = x + attn_out
        x_ln2 = self.norm2(x)
        x = x + self.mlp(x_ln2)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: Optional[int] = None, dim: int = 512, n_layers: int = 6, n_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.0, pad_token_id: int = 0,
                 output_is_emb: bool = False, pool_output: bool = False, 
                 input_is_emb: bool = False):
        """Transformer Encoder block

        Modes:
        - token IDs in, pooled embedding out: input_is_emb=False, pool_output=True
        - token IDs in, token logits out: input_is_emb=False, pool_output=False, output_is_emb=False
        - embeddings in, embeddings out: input_is_emb=True, output_is_emb=True

        vocab_size is required only when a token embedding layer or a logits head is used.
        """
        super().__init__()
        assert dim % n_heads == 0
        self.pool_output = pool_output
        self.input_is_emb = input_is_emb
        self.output_is_emb = output_is_emb

        needs_token_emb = not self.input_is_emb
        needs_lm_head = (not self.pool_output) and (not self.output_is_emb)
        if (needs_token_emb or needs_lm_head) and vocab_size is None:
            raise ValueError(
                "vocab_size must be set when token embedding or logits head is enabled."
            )

        if needs_token_emb:
            # For token-ID input we need a token embedding table.
            self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        else:
            self.token_emb = None

        self.pos_dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(dim // n_heads)
        self.layers = nn.ModuleList([])
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)

        if needs_lm_head:
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)
            nn.init.xavier_uniform_(self.lm_head.weight)
            # optional weight tying:
            # only use when not input_is_emb
            #self.lm_head.weight = self.token_emb.weight
        else:
            self.lm_head = None

    def forward(self, input: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                is_causal: Optional[bool]=False):
        if not self.input_is_emb:
            x = self.token_emb(input)
        else:
            x = input
        b, seq, feat = x.shape
        x = self.pos_dropout(x)

        if attention_mask is None:
            attention_mask = torch.ones((b, seq), dtype=torch.bool, device=x.device)
        else:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.to(torch.bool)
            else:
                raise ValueError("attention_mask must be None or shape (batch, seq)")

        sdpa_mask = attention_mask[:, None, None, :]

        for layer in self.layers:
            x = layer(x, attn_mask=sdpa_mask, rotary=self.rotary, is_causal=is_causal)

        x = self.norm(x)

        if self.pool_output:
            # return pooled final embs
            mask = attention_mask.unsqueeze(-1)
            summed = (x * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            pooled = summed / denom
            return F.normalize(pooled, p=2, dim=-1)
        elif self.output_is_emb:
            # just return final embs
            return x
        else:
            # return logits
            return self.lm_head(x)

def compute_mlm_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100):
    """For testing"""
    assert logits.dim() == 3, "logits must be (b, seq, vocab) for MLM loss"
    vocab_size = logits.size(-1)
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
    return loss_fct(logits.view(-1, vocab_size), labels.view(-1))

if __name__ == "__main__":
    """Run this to verify the implementation."""
    import time
    batch = 2
    seq = 16
    vocab = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    model = TransformerEncoder(vocab_size=vocab, dim=256, n_layers=4, n_heads=8, mlp_ratio=2.0).to(device)
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_parameters} parameters")
    input_ids = torch.randint(0, vocab - 1, (batch, seq)).to(device)
    attention_mask = (input_ids != 0).to(device)
    labels = input_ids.clone()
    rand = torch.rand(labels.shape)
    mask_positions = rand < 0.15
    labels[~mask_positions] = -100
    mask_token_id = vocab - 1
    input_ids[mask_positions] = mask_token_id
    start_t = time.time()
    logits = model(input_ids, attention_mask=attention_mask, is_causal=True)
    loss = compute_mlm_loss(logits, labels)

    print("time taken:", time.time()-start_t)
    print("loss", loss.item())

    #----------------------------
    # Test Configurations
    #----------------------------

    batch_size = 4
    sequence_length = 8
    n_vocab = 256
    feat_dim = 32
    model_dim = 32
    n_layers = 4
    n_heads = 4
    mlp_ratio = 2.0

    # with input_is_emb, shapes need to match
    assert model_dim == feat_dim
    input_emb = torch.rand((batch_size, sequence_length, feat_dim))
    input_ids = torch.randint(0, n_vocab - 1, (batch_size, sequence_length))
    attention_mask = torch.randint(0, 1, (batch_size, sequence_length), dtype=torch.bool)

    #----------------------------
    # embedding in, logits out
    #----------------------------
    model = TransformerEncoder(vocab_size=n_vocab, 
                            dim=model_dim, 
                            n_layers=n_layers, 
                            n_heads=n_heads, 
                            mlp_ratio=mlp_ratio, 
                            input_is_emb=True,
                            pool_output=False)

    output = model(input_emb)
    # since we dont pool output shape should be vocab size
    assert output.shape[-1] == n_vocab
    # test causal
    output = model(input_emb, is_causal=True)
    # test masked language learning
    output = model(input_emb, attention_mask=attention_mask)

    #----------------------------
    # embedding in, embedding out
    #----------------------------
    model = TransformerEncoder(vocab_size=n_vocab, 
                            dim=model_dim, 
                            n_layers=n_layers, 
                            n_heads=n_heads, 
                            mlp_ratio=mlp_ratio, 
                            input_is_emb=False,
                            pool_output=True)

    output = model(input_ids)
    # pooled output should be same as feat size in and modal dim
    assert output.shape[-1] == feat_dim == model_dim
    # test causal
    output = model(input_ids, is_causal=True)
    # test masked language learning
    output = model(input_ids, attention_mask=attention_mask)

    #----------------------------
    # IDs in, logits out
    #----------------------------
    model = TransformerEncoder(vocab_size=n_vocab, 
                            dim=model_dim, 
                            n_layers=n_layers, 
                            n_heads=n_heads, 
                            mlp_ratio=mlp_ratio, 
                            input_is_emb=False,
                            pool_output=False)

    output = model(input_ids)
    # since we dont pool output shape should be vocab size
    assert output.shape[-1] == n_vocab
    # test causal
    output = model(input_ids, is_causal=True)
    # test masked language learning
    output = model(input_ids, attention_mask=attention_mask)

    #----------------------------
    # IDs in, embeddings out
    #----------------------------
    model = TransformerEncoder(vocab_size=n_vocab, 
                            dim=model_dim, 
                            n_layers=n_layers, 
                            n_heads=n_heads, 
                            mlp_ratio=mlp_ratio, 
                            input_is_emb=False,
                            pool_output=True)

    output = model(input_ids)
    # pooled output should be same as feat size in and modal dim
    assert output.shape[-1] == feat_dim == model_dim
    # test causal
    output = model(input_ids, is_causal=True)
    # test masked language learning
    output = model(input_ids, attention_mask=attention_mask)

    #----------------------------
    # Embs in, raw embeddings out (not pooled)
    #----------------------------
    model = TransformerEncoder(vocab_size=n_vocab, 
                            dim=model_dim, 
                            n_layers=n_layers, 
                            n_heads=n_heads, 
                            mlp_ratio=mlp_ratio, 
                            input_is_emb=True,
                            output_is_emb=True, #return embs
                            pool_output=False) # but dont pool

    output = model(input_emb)
    # pooled output should be same as feat size in and modal dim
    assert output.shape[-1] == feat_dim == model_dim
    assert len(output.shape) == 3, "output should be (batch, seq, feat) when output_is_emb is True, and not pooled"