```
                                    ┌─────────────────────────────────────────────────────────────────┐
                                    │                        DeepSeek-V3 Transformer                  │
                                    └─────────────────────────────────────────────────────────────────┘
                                                                  输入
                                                                    ↓
                                    ┌─────────────────────────────────────────────────────────────────┐
                                    │                    Parallel Embedding Layer                     │
                                    └─────────────────────────────────────────────────────────────────┘
                                                                    ↓
                                    ┌─────────────────────────────────────────────────────────────────┐
                                    │          Rotary Position Encoding( (RoPE with YaRN) )           │
                                    └─────────────────────────────────────────────────────────────────┘
                                                                    ↓
                                    ┌─────────────────────────────────────────────────────────────────┐
                                    │                    Transformer Blocks (× n_layers)              │
                                    ├─────────────────────────────────────────────────────────────────┤
                                    │  ┌────────────────────────────────────────────────────────────┐ │
                                    │  │                    Transformer Block                       │ │
                                    │  │  ┌─────────────────────────────────────────────────────┐   │ │
                                    │  │  │              Residual Connection 1                  │   │ │
                                    │  │  └─────────────────────────────────────────────────────┘   │ │
                                    │  │                            ↓                               │ │
                                    │  │  ┌─────────────────────────────────────────────────────┐   │ │
                                    │  │  │                  RMSNorm (Attention)                │   │ │
                                    │  │  └─────────────────────────────────────────────────────┘   │ │
                                    │  │                            ↓                               │ │
                                    │  │  ┌─────────────────────────────────────────────────────┐   │ │
                                    │  │  │          Multi-Head Latent Attention (MLA)          │   │ │
                                    │  │  │  ┌──────────────────────────────────────────────────┐   │ │
                                    │  │  │  │                      Query Projection            │   │ │
                                    │  │  │  │      ┌─────────┐  ┌─────────┐  ┌─────────┐       │   │ │
                                    │  │  │  │      │ LoRA A  │→ │ RMSNorm │→ │ LoRA B  │       │   │ │
                                    │  │  │  │      └─────────┘  └─────────┘  └─────────┘       │   │ │
                                    │  │  │  └──────────────────────────────────────────────────┘   │ │
                                    │  │  │  ┌──────────────────────────────────────────────────┐   │ │
                                    │  │  │  │                    Key-Value Projection          │   │ │
                                    │  │  │  │        wkv_a → [kv, k_pe] → RMSNorm → wkv_b      │   │ │
                                    │  │  │  │                    ↓         ↓                   │   │ │
                                    │  │  │  │            [k_nope, v]    k_pe(rotary)           │   │ │
                                    │  │  │  └──────────────────────────────────────────────────┘   │ │
                                    │  │  │  ┌──────────────────────────────────────────────────┐   │ │
                                    │  │  │  │                  Attention Computation            │  │ │
                                    │  │  │  │    scores = (q_nope·k_nope + q_pe·k_pe) / √d      │  │ │
                                    │  │  │  │    attn_weights = softmax(scores + mask)          │  │ │
                                    │  │  │  │    output = attn_weights · v                      │  │ │
                                    │  │  │  └───────────────────────────────────────────────────┘  │ │
                                    │  │  │  ┌───────────────────────────────────────────────────┐  │ │
                                    │  │  │  │                    Output Projection              │  │ │
                                    │  │  │  └───────────────────────────────────────────────────┘  │ │
                                    │  │  └──────────────────────────────────────────────────────┘  │ │
                                    │  │                            ↓                               │ │
                                    │  │  ┌─────────────────────────────────────────────────────┐   │ │
                                    │  │  │              Residual Connection 2                  │   │ │
                                    │  │  └─────────────────────────────────────────────────────┘   │ │
                                    │  │                            ↓                               │ │
                                    │  │  ┌─────────────────────────────────────────────────────┐   │ │
                                    │  │  │                  RMSNorm (FFN)                      │   │ │
                                    │  │  └─────────────────────────────────────────────────────┘   │ │
                                    │  │                            ↓                               │ │
                                    │  │  ┌─────────────────────────────────────────────────────┐   │ │
                                    │  │  │        FeedForward Network (Dense or MoE)           │   │ │
                                    │  │  │      ┌───────────────────┐  ┌─────────────────┐     │   │ │
                                    │  │  │      │   Dense Layers    │  │   MoE Layers    │     │   │ │
                                    │  │  │      │  (n_dense_layers) │  │  (剩余layers)   │     │   │ │
                                    │  │  │      │                   │  │                 │     │   │ │
                                    │  │  │      │  MLP:             │  │  MoE:           │     │   │ │
                                    │  │  │      │  - w1(SiLU)       │  │  - Gate路由     │     │   │ │
                                    │  │  │      │  - w3(linear)     │  │  - Expert计算   │     │   │ │
                                    │  │  │      │  - w2(project)    │  │  - Shared专家   │     │   │ │
                                    │  │  │      │                   │  │                 │     │   │ │
                                    │  │  │      └───────────────────┘  └─────────────────┘     │   │ │
                                    │  │  └─────────────────────────────────────────────────────┘   │ │
                                    │  │                            ↓                               │ │
                                    │  │  ┌─────────────────────────────────────────────────────┐   │ │
                                    │  │  │                      output                         │   │ │
                                    │  │  └─────────────────────────────────────────────────────┘   │ │
                                    │  └────────────────────────────────────────────────────────────┘ │
                                    │                            × n_layers                           │
                                    └─────────────────────────────────────────────────────────────────┘
                                                                    ↓
                                    ┌─────────────────────────────────────────────────────────────────┐
                                    │                      Final RMSNorm                              │
                                    └─────────────────────────────────────────────────────────────────┘
                                                                    ↓
                                    ┌─────────────────────────────────────────────────────────────────┐
                                    │                    Output Projection                            │
                                    └─────────────────────────────────────────────────────────────────┘
                                                                    ↓
                                                                  输出
```



*** 

## 门控

### 计算思路

计算每个token对每个专家的分数（可能有偏置）-> 分组路由 -> topk选取k个专家

- 例子：

  没有分组的路由 (n_expert_groups = 1)：输入token → 计算128个专家分数 → Top-10选择 → 使用10个专家
  
  有分组的路由 (n_expert_groups = 8, n_limited_groups = 2)：输入token → 计算128个专家分数 → 分成8组，每组16个专家 → 计算每组得分 (取组内前2个专家分数和) → Top-2选择 (选择得分最高的2个组) → 在选中的2个组(共32个专家)中做Top-10选择 → 使用10个专家


  

<details>
<summary>Gate代码实现</summary>
  
```python
class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices
```
</details>




## 专家
- 每个专家都是llama的前馈网

  


## MOE

- 将路由专家的结果与共享专家合并
- 共享专家：
  - 是MoE层中所有token都会经过的专家，与路由专家（需要门控选择）形成互补。

<details>
<summary>MOE逻辑实现(非完整代码）</summary>

```python
class MoE(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 路由专家计算 (选择性)
        weights, indices = self.gate(x)  # 选择部分专家
        y = torch.zeros_like(x)
        
        # 只计算被选中的路由专家
        for i in selected_experts:
            y += self.routed_experts[i](x) * weights[i]
        
        # 2. 共享专家计算 (所有token都经过)
        z = self.shared_experts(x)  # 所有token都计算
        
        # 3. 合并输出
        return y + z  # 路由专家输出 + 共享专家输出
```
</details>




## MLA



### 原理

当使用 LoRA 时，查询投影为
\[
Q = W_q^B \cdot \text{Norm}(W_q^A \cdot X)
\]

查询拆分为非位置部分和旋转位置部分：Q = [Q_{\text{nope}}; Q_{\text{rope}}]
​键拆分为非位置部分和旋转位置部分：K = [K_{\text{nope}}; K_{\text{rope}}]

注意力分数计算
\[
\text{scores} = \frac{1}{\sqrt{d_{qk}}} \left( Q_{\text{nope}} K_{\te
\]
​
输出：\text{output} = W_o \cdot (\text{softmax}(\text{scores}) \cdot V)



111111111111111111111111111111111111

<details>
<summary>MLA实现</summary>

```python
class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args: ModelArgs):
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        if attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x

```
</details>















***


## 其他优化

### ParallelEmbedding (分布式词嵌入，支持模型并行)

<details>
<summary>ParallelEmbedding实现</summary>

```python
class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y
```
</details>


### Rotary Position Encoding (RoPE with YaRN) precompute_freqs_cis() - 支持长度外推的动态缩放


### 量化 FP8/BF16 混合精度



















