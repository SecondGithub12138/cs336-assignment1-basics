from __future__ import annotations

import os
import time
import numpy.typing as npt
import torch
import regex
import multiprocessing
import functools
import math
import pickle
import numpy
from typing import Optional
from einops import rearrange
from collections.abc import Iterable, Iterator, Callable
from typing import IO, Any, BinaryIO
from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections import defaultdict, Counter
from cs336_basics.pretokenization import find_chunk_boundaries

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        d_in (int): The size of the input dimension
        d_out (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The input tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    linear = Linear(d_in, d_out)
    with torch.no_grad():
        linear.weight.copy_(weights)
    return linear(in_features)
    
class Linear(torch.nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        """
        Args:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on 
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        W = torch.empty(d_out, d_in, device=device, dtype=dtype)
        mean = 1
        std = math.sqrt(2/(d_in + d_out))
        torch.nn.init.trunc_normal_(W, mean, std, -3*std, 3*std)
        self.weight = torch.nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
        

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding = Embedding(vocab_size, d_model)
    with torch.no_grad():
        embedding.weight.copy_(weights)
    return embedding(token_ids)

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        W = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        mean = 0
        std = 1
        torch.nn.init.trunc_normal_(W, mean, std, -3*std, 3*std)
        self.weight = torch.nn.Parameter(W)
        

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids]

class SWIGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # Linear layers: input shape -> output shape
        # self.w1 = Linear(d_model, d_ff)  # d_model -> d_ff
        # self.w2 = Linear(d_ff, d_model)  # d_ff -> d_model
        # self.w3 = Linear(d_model, d_ff)  # d_model -> d_ff
        self.w1 = torch.nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = torch.nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = torch.nn.Parameter(torch.empty(d_ff, d_model))

    def forward(self, x: Float[Tensor, "... d_model"]):
        # SwiGLU formula: (Swish(x @ W1) * (x @ W3)) @ W2
        # where Swish(x) = x * sigmoid(x)
        gate = x @ self.w1.T  # shape: (... d_ff)
        gate_activated = gate * torch.sigmoid(gate)  # Swish activation
        value = x @ self.w3.T  # shape: (... d_ff)
        return (gate_activated * value) @ self.w2.T # shape: (... d_model)

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    swiglu = SWIGLU(d_model, d_ff)
    with torch.no_grad():
        swiglu.w1.copy_(w1_weight)
        swiglu.w2.copy_(w2_weight)
        swiglu.w3.copy_(w3_weight)
    return swiglu(in_features)
    
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    raise NotImplementedError

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    to_be_softmax = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    if mask is not None:
        to_be_softmax = to_be_softmax.masked_fill(mask==False, float("-inf"))
    return softmax(to_be_softmax, -1) @ V

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    return scaled_dot_product_attention(Q, K, V, mask)

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 0,
    theta: float = 0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len #for rope
        self.theta = theta #for rope
        self.rope = RoPE(self.theta, self.d_k, self.max_seq_len, device=None)
        self.q_proj_weight = torch.nn.Parameter(torch.empty(d_model, d_model))
        self.k_proj_weight = torch.nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj_weight = torch.nn.Parameter(torch.empty(d_model, d_model))
        self.o_proj_weight = torch.nn.Parameter(torch.empty(d_model, d_model))
    
    def forward(self, in_features: Float[Tensor, " ... sequence_length d_in"], mask: Bool[Tensor, " ... queries keys"] | None = None, token_positions: Int[Tensor, " ... sequence_length"] | None = None) -> Float[Tensor, " ... queries d_v"]:
        qkv_big_weight = torch.cat([self.q_proj_weight, self.k_proj_weight, self.v_proj_weight])
        qkv = in_features @ qkv_big_weight.T
        q, k, v = qkv.chunk(3, -1)
        q = rearrange(q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        k = rearrange(k, "... seq (h d) -> ... h seq d", h=self.num_heads)
        v = rearrange(v, "... seq (h d) -> ... h seq d", h=self.num_heads)
        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        result = scaled_dot_product_attention(q, k, v, mask)
        concat = rearrange(result, "... h seq d -> ... seq (h d)")
        output = concat @ self.o_proj_weight.T
        return output


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    ma = MultiheadSelfAttention(d_model, num_heads)
    with torch.no_grad():
        ma.q_proj_weight.copy_(q_proj_weight)
        ma.k_proj_weight.copy_(k_proj_weight)
        ma.v_proj_weight.copy_(v_proj_weight)
        ma.o_proj_weight.copy_(o_proj_weight)
    seq_len = in_features.shape[-2]
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    return ma(in_features, mask)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    ma = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta)
    with torch.no_grad():
        ma.q_proj_weight.copy_(q_proj_weight)
        ma.k_proj_weight.copy_(k_proj_weight)
        ma.v_proj_weight.copy_(v_proj_weight)
        ma.o_proj_weight.copy_(o_proj_weight)
    seq_len = in_features.shape[-2]
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    return ma(in_features, mask, token_positions)

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        freqs = theta ** (-torch.arange(0, d_k, 2).float() / d_k)
        freqs = freqs.unsqueeze(0)
        positions = torch.arange(max_seq_len).unsqueeze(1)
        angles = freqs * positions
        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        result_even = x_even * cos - x_odd * sin
        result_odd = x_even * sin + x_odd * cos
        result = torch.empty_like(x)
        result[..., ::2] = result_even
        result[..., 1::2] = result_odd
        return result

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RoPE(theta, d_k, max_seq_len, device=None)
    return rope(in_query_or_key, token_positions)

def softmax(x: torch.Tensor, dim: int):
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max
    x_exp = torch.exp(x_stable)
    return x_exp/torch.sum(x_exp, dim=dim, keepdim=True)

class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, Tensor],
    ):
        super().__init__()
        # multi head attention prep
        self.rmsnorm1 = RMSNorm(d_model)
        self.mha = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta)
        with torch.no_grad():
            self.rmsnorm1.weight.copy_(weights["ln1.weight"])
            self.mha.q_proj_weight.copy_(weights["attn.q_proj.weight"])
            self.mha.k_proj_weight.copy_(weights["attn.k_proj.weight"])
            self.mha.v_proj_weight.copy_(weights["attn.v_proj.weight"])
            self.mha.o_proj_weight.copy_(weights["attn.output_proj.weight"])
        
        # FFN prep
        self.rmsnorm2 = RMSNorm(d_model)
        self.swiglu = SWIGLU(d_model, d_ff)
        with torch.no_grad():
            self.rmsnorm2.weight.copy_(weights["ln2.weight"])
            self.swiglu.w1.copy_(weights["ffn.w1.weight"])
            self.swiglu.w2.copy_(weights["ffn.w2.weight"])
            self.swiglu.w3.copy_(weights["ffn.w3.weight"])
    
    def forward(self, in_features: Float[Tensor, " batch sequence_length d_model"]) -> Float[Tensor, " batch sequence_length d_model"]:
        seq_len = in_features.shape[-2]
        # mask
        self.mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device))
        # position
        self.token_positions = torch.arange(seq_len, device=in_features.device)
        in_features = in_features + self.mha(self.rmsnorm1(in_features), self.mask, token_positions=self.token_positions)
        return in_features + self.swiglu(self.rmsnorm2(in_features))

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    tb = TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, weights)
    return tb(in_features)

    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    # encoding 
    eb = Embedding(vocab_size, d_model)
    with torch.no_grad():
        eb.weight.copy_(weights["token_embeddings.weight"])
    in_features = eb(in_indices)
    # transformer
    for i in range(num_layers):
        layer_weights = {}
        layer_weights["attn.q_proj.weight"] = weights[f"layers.{i}.attn.q_proj.weight"]
        layer_weights["attn.k_proj.weight"] = weights[f"layers.{i}.attn.k_proj.weight"]
        layer_weights["attn.v_proj.weight"] = weights[f"layers.{i}.attn.v_proj.weight"]
        layer_weights["attn.output_proj.weight"] = weights[f"layers.{i}.attn.output_proj.weight"]
        layer_weights["ln1.weight"] = weights[f"layers.{i}.ln1.weight"]
        layer_weights["ln2.weight"] = weights[f"layers.{i}.ln2.weight"]
        layer_weights["ffn.w1.weight"] = weights[f"layers.{i}.ffn.w1.weight"]
        layer_weights["ffn.w2.weight"] = weights[f"layers.{i}.ffn.w2.weight"]
        layer_weights["ffn.w3.weight"] = weights[f"layers.{i}.ffn.w3.weight"]
        transformer_block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len=context_length, theta=rope_theta, weights=layer_weights)
        in_features = transformer_block(in_features)
    
    # Last Norm
    rmsnorm = RMSNorm(d_model)
    with torch.no_grad():
        rmsnorm.weight.copy_(weights["ln_final.weight"])
    in_features = rmsnorm(in_features)
    
    # Linear
    linear = Linear(d_model, vocab_size)
    with torch.no_grad():
        linear.weight.copy_(weights["lm_head.weight"])
    in_features = linear(in_features)
    # return softmax(in_features, -1)
    return in_features
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    model = RMSNorm(d_model, eps)
    with torch.no_grad():
        model.weight.copy_(weights)
    return model(in_features)


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        W = torch.empty(d_model, device=device, dtype=dtype)
        self.weight = torch.nn.Parameter(W)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        # Upcast to float32 for numerical stability
        original_dtype = x.dtype
        x_float32 = x.to(torch.float32)

        # Compute RMS in float32
        rms = torch.sqrt(x_float32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normalized = (x_float32 / rms).to(original_dtype)

        # Apply affine transform
        return normalized * self.weight


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    samples = numpy.random.randint(low = 0, high = dataset.size - context_length, size = (batch_size,))
    sequences = numpy.stack([dataset[s : s + context_length] for s in samples])
    labels = numpy.stack([dataset[s + 1 : s + context_length + 1] for s in samples])
    return (torch.from_numpy(sequences).long().to(device), torch.from_numpy(labels).long().to(device))
    
def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return softmax(in_features, dim)

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    # Method 1
    rows_idx = torch.arange(targets.shape[0])
    # ----
    # probs = softmax(inputs, -1)
    # log_max = -torch.log(probs[rows_idx, targets])
    inputs = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    log_max = inputs[rows_idx, targets] - torch.log(torch.sum(torch.exp(inputs), dim=-1))
    # ----
    return -torch.mean(log_max)
    # Method 2
    # probs = softmax(inputs, -1)
    # batch_indices = torch.arange(targets.shape[0], device=inputs.device)
    # correct_probs = probs[batch_indices, targets]
    # return -torch.mean(torch.log(correct_probs))
    # Method 3
    # log_probs = torch.log_softmax(inputs, dim=-1)
    # batch_indices = torch.arange(targets.shape[0], device=inputs.device)
    # return -torch.mean(log_probs[batch_indices, targets])

def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    return cross_entropy(inputs, targets)

@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    params_with_grad = [p for p in parameters if p.grad is not None]
    # l2 = torch.sqrt(sum(torch.sum(p.grad))for p in params_with_grad)
    total_norm = 0
    for param in params_with_grad:
        total_norm += torch.sum(torch.pow(param.grad, 2))
    
    if torch.sqrt(total_norm) > max_l2_norm:
        for param in params_with_grad:
            param.grad *= max_l2_norm/(torch.sqrt(total_norm) + 1e-6)

def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    gradient_clipping(parameters, max_l2_norm)

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.

                oldm = state.get("exp_avg", torch.zeros_like(p))
                m = beta1 * oldm + (1 - beta1) * p.grad

                oldv = state.get("exp_avg_sq", torch.zeros_like(p))
                v = beta2 * oldv + (1 - beta2) * p.grad * p.grad

                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                p -= lr_t * m / (torch.sqrt(v) + eps)
                p -= lr * weight_decay * p # Update weight tensor in-place.

                state["t"] = t + 1 # Increment iteration number
                state["exp_avg"] = m
                state["exp_avg_sq"] = v
        return loss

def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW

def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it/warmup_iters*max_learning_rate
    elif it > cosine_cycle_iters:
        return min_learning_rate
    else:
        return min_learning_rate + 1/2 * (1 + math.cos((it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters)) ) * (max_learning_rate - min_learning_rate)



def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    return lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    if isinstance(out, (str or os.PathLike)):
        with open(out, "wb") as f:
            torch.save(checkpoint, f)
    else:
        torch.save(checkpoint, out)
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    if isinstance(src, (str | os.PathLike)):
        with open(src, "rb") as f:
            checkpoint = torch.load(f)
    else:
        checkpoint = torch.load(src)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)

def process_chunk(input_path: str, special_tokens: list[str], chunk: tuple[int]) -> Counter:
    with open(input_path, "rb") as f:
        f.seek(chunk[0])
        chunk_text = f.read(chunk[1] - chunk[0]).decode("utf-8")
    
    counter: dict[tuple[bytes], int] = Counter()
    # Strip lefted special_token leaved by find_chunk_boundaries().
    # Hopefully there should be only 1 new chunk as chunk = new_chunk + special_token
    special_tokens_str = "|".join(map(regex.escape, special_tokens))
    for new_chunk in regex.split(special_tokens_str, chunk_text):
        # print out new_chunk and chunk
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for word in regex.finditer(PAT, new_chunk):
            word_tuple = tuple(bytes([w]) for w in word.group().encode("utf-8"))
            counter[word_tuple] += 1
    return counter

def _print_timing_summary(phase_times: dict[str, float], vocab_size: int, num_merges: int) -> None:
    """Print detailed timing breakdown of BPE training phases."""
    print(f"         ✓ Completed in {phase_times['merging_total']:.2f}s")
    print(f"           - Pair counting: {phase_times['merging_pair_counting']:.2f}s ({100*phase_times['merging_pair_counting']/phase_times['merging_total']:.1f}%)")
    print(f"           - Merge updates: {phase_times['merging_update']:.2f}s ({100*phase_times['merging_update']/phase_times['merging_total']:.1f}%)")
    print(f"         Final vocab size: {vocab_size:,} tokens ({num_merges:,} merges)")

    # Print timing summary
    print("   [4/4] Training complete!")
    total_time = sum(phase_times.values())
    print(f"\n   ⏱️  Detailed timing breakdown:")
    print(f"         - Vocab init:        {phase_times['vocab_init']:8.2f}s ({100*phase_times['vocab_init']/total_time:5.1f}%)")
    print(f"         - Pre-tokenization:  {phase_times['pre_tokenization']:8.2f}s ({100*phase_times['pre_tokenization']/total_time:5.1f}%)")
    print(f"         - BPE merging:       {phase_times['merging_total']:8.2f}s ({100*phase_times['merging_total']/total_time:5.1f}%)")
    print(f"           ├─ Pair counting:  {phase_times['merging_pair_counting']:8.2f}s ({100*phase_times['merging_pair_counting']/total_time:5.1f}%)")
    print(f"           └─ Merge updates:  {phase_times['merging_update']:8.2f}s ({100*phase_times['merging_update']/total_time:5.1f}%)")
    print(f"         ─────────────────────────────────")
    print(f"         Total:               {total_time:8.2f}s (100.0%)")


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # Timing setup
    phase_times = {}

    # Init vocab
    print("   [1/4] Initializing vocabulary...")
    t0 = time.time()
    vocab: dict[int, bytes] = {i : bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    phase_times["vocab_init"] = time.time() - t0
    print(f"         ✓ Completed in {phase_times['vocab_init']:.2f}s")

    # Pre-tokenization
    print("   [2/4] Pre-tokenization and counting...")
    t0 = time.time()
    # =============================================Multi Process=============================================
    # Seperate full text to documents and process Note: don't load the whole text into memory
    words_counter: dict[tuple[bytes], int] = Counter()
    with open(input_path, "rb") as f:
        num_process = 4 
        boundaries = find_chunk_boundaries(f, num_process, b"<|endoftext|>")
        with multiprocessing.Pool(processes=num_process) as pool:
            counters = pool.imap_unordered(functools.partial(process_chunk, input_path, special_tokens), zip(boundaries[:-1], boundaries[1:]))
            for counter in counters:
                words_counter.update(counter)
    # =============================================Single Process=============================================
    # GPT-2 pre-tokenization pattern
    # def to_bytes_tuple(word: str) -> tuple[bytes]:
    #     encoded = word.encode("utf-8")
    #     return tuple(bytes([e]) for e in encoded)
    

    # PAT = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    # with open(input_path, "r", encoding="utf-8") as f:
    #     text = f.read()
    # # print(f"| Original special_tokens: {special_tokens}")
    # chunks = regex.split("|".join(map(regex.escape, special_tokens)), text)

    # words_counter = defaultdict(int)
    # for chunk in chunks:
    #     for match in regex.finditer(PAT, chunk):
    #         words_counter[to_bytes_tuple(match.group())] += 1   # key of pre_tokens_cnt e.g. (b'H', b'e', b'l', b'l', b'o')
    # =============================================Single Process(End)=============================================
    # Repeat for at most vocab_size time or when there is nothing to merge
    phase_times["pre_tokenization"] = time.time() - t0
    print(f"         ✓ Completed in {phase_times['pre_tokenization']:.2f}s")
    print(f"         Found {len(words_counter):,} unique words")
    print("   [3/4] BPE merging iterations...")
    t0 = time.time()
    merges = []

    # Track time for different parts of merge loop
    time_pair_counting = 0
    time_merge_update = 0

    while len(vocab) < vocab_size:
        # Counter "byte" count globally
        t_pairs = time.time()
        pairs_counter: dict[tuple[bytes], int] = Counter()
        for word, count in words_counter.items():
            for i in range(len(word) - 1):
                pairs_counter[(word[i], word[i + 1])] += count
        if not pairs_counter:
            break
        # Pick up the byte pair from candidates for merge, e.g. to_be_merge: (b'a, b'b)
        to_be_merge = max(pairs_counter.items(), key = lambda x : (x[1], x[0][0], x[0][1]))[0]
        time_pair_counting += time.time() - t_pairs

        # e.g b'ab OR b'\x61\x62'
        t_update = time.time()
        new_token = to_be_merge[0] + to_be_merge[1]
        merges.append(to_be_merge)
        vocab[len(vocab)] = new_token
        # Start merge. e.g. word: (b'a, b'b, c'c) OR (b'ab, c'c)
        to_be_replace = []
        for word, count in words_counter.items():
            updated = False
            new_word_list = []
            i = 0
            while i < len(word):
                if word[i: i + 2] == to_be_merge:
                    new_word_list.append(new_token)
                    i += 2
                    updated = True
                else:
                    new_word_list.append(word[i])
                    i += 1
            if updated:
                to_be_replace.append((word, tuple(new_word_list), count))
        for old_word_tuple, new_word_tuple, count in to_be_replace:
            del words_counter[old_word_tuple]
            words_counter[new_word_tuple] = count + words_counter.get(new_word_tuple, 0)
        time_merge_update += time.time() - t_update

        # Progress update every 1000 merges
        if len(merges) % 1000 == 0:
            print(f"         Progress: {len(vocab):,}/{vocab_size:,} tokens ({len(merges):,} merges)")

    phase_times["merging_total"] = time.time() - t0
    phase_times["merging_pair_counting"] = time_pair_counting
    phase_times["merging_update"] = time_merge_update
    # Print timing summary
    _print_timing_summary(phase_times, len(vocab), len(merges))

    return (vocab, merges)

class Tokenizer:
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        if special_tokens is None:
            special_tokens = []
        return cls(vocab, merges, special_tokens)
        
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.merges_priorities: dict[tuple[bytes, bytes], int] = dict(zip(merges, range(len(merges))))
        self.re_vocab: dict[bytes, int] = {v : k for k, v in vocab.items()}
        for s_tokens in self.special_tokens:
            if s_tokens.encode("utf-8") not in self.re_vocab:
                token_id = len(self.vocab)
                self.vocab[token_id] = s_tokens.encode("utf-8")
                self.re_vocab[s_tokens.encode("utf-8")] = token_id

    def encode(self, text: str) -> list[int]:
        # handle special tokens first
        res: list[int] = []
        sort_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern ="|".join(map(regex.escape, sort_special_tokens))
        if pattern:
            chunks = regex.split(f"({pattern})", text) 
        else:
            chunks = [text]
        for chunk in chunks:
            if chunk in sort_special_tokens:
                res.append(self.re_vocab[chunk.encode("utf-8")])
            else:
                # encode the normal splited chunk
                PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                for word in regex.finditer(PAT, chunk):
                    word_tuple = tuple(bytes([c]) for c in word.group().encode("utf-8"))
                    while True:
                        new_word_list = []
                        if len(word_tuple) <= 1:
                            break
                        update_idx = min(range(len(word_tuple) - 1), key=lambda i : (self.merges_priorities.get(word_tuple[i: i + 2], math.inf)))
                        if word_tuple[update_idx: update_idx + 2] not in self.merges_priorities:
                            break
                        idx = 0
                        while idx < len(word_tuple):
                            if idx == update_idx and idx + 1 < len(word_tuple):
                                new_word_list.append(word_tuple[idx] + word_tuple[idx+1])
                                idx += 2
                            else:
                                new_word_list.append(word_tuple[idx])
                                idx += 1
                        word_tuple = tuple(new_word_list)
                    res.extend(self.re_vocab[word_byte] for word_byte in word_tuple)
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for str in iterable:
            for each in self.encode(str):
                yield each
    # usage: for token_id in tokenizer.encode(huge_text):
    # process(token_id)

    def decode(self, ids: list[int]) -> str:
        res = b""
        for token_id in ids:
            res += self.vocab[token_id]
        return res.decode("utf-8", errors="replace")

    def encode_large_file(
        self,
        file_path: str,
        num_chunks: int = 4,
        special_token_separator: bytes = b"<|endoftext|>"
    ) -> Iterator[int]:
        """
        逐块读取大文件并编码，确保special token不被分割

        Args:
            file_path: 文件路径
            num_chunks: 分块数量（用于控制每次读取的大小）
            special_token_separator: 用于分割的special token（作为chunk边界）

        Yields:
            int: 每个token ID
        """
        def chunk_generator():
            with open(file_path, "rb") as f:
                # 找到不会分割special token的边界
                boundaries = find_chunk_boundaries(f, num_chunks, special_token_separator)

                # 逐个chunk处理
                for i in range(len(boundaries) - 1):
                    start, end = boundaries[i], boundaries[i + 1]
                    f.seek(start)
                    chunk_bytes = f.read(end - start)
                    chunk_text = chunk_bytes.decode("utf-8")
                    yield chunk_text

        # 使用 encode_iterable 来处理每个chunk
        for token_id in self.encode_iterable(chunk_generator()):
            yield token_id









    
