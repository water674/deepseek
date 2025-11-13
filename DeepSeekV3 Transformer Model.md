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



*** 门控


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





