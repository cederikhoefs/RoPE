import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_geometric.utils import to_dense_batch

from .performer_layer import FastAttention


def init_omega_matrix(omega_matrix, init_omega: str, d: int, k: int):
    """
    Initialize an Omega matrix according to the specified strategy.
    
    Args:
        omega_matrix: The nn.Linear layer to initialize
        init_omega: Initialization strategy
        d: Total feature dimension 
        k: Rotational feature dimension
    """
    with torch.no_grad():
        match init_omega:
            case "zero":
                nn.init.zeros_(omega_matrix.weight)
            
            case "exponential":
                # Initialize with random frequencies
                rand_freqs = torch.rand(d//2, k, device=omega_matrix.weight.device)
                # Apply exponential decay
                decay_factors = torch.tensor([[10000**(2*i/(d//2)) for _ in range(k)] 
                                        for i in range(d//2)], device=omega_matrix.weight.device)
                omega_matrix.weight.copy_(rand_freqs / decay_factors)
            
            case "uniform":
                pass

            case "orthogonal":
                nn.init.orthogonal_(omega_matrix.weight)

            case "none":
                nn.init.eye_(omega_matrix.weight)


def rotate(x, sin, cos):
    x1, x2 = x[..., ::2], x[..., 1::2] # (..., d//2)
    x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1) # (..., d//2, 2)
    return x_rotated.flatten(-2) # (..., d)

class GraphRoPE(nn.Module):
    def __init__(self, 
                 k: int, 
                 d: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 enable: bool = True,
                 init_omega: str = "zero",
                 attn_type: str = "Full",
                 shared_omega_q: nn.Module = None,
                 shared_omega_k: nn.Module = None,
                 double_omega: bool = False,
                 freeze_omega: bool = False,
                 return_logits: bool = False):
        """
        Multi-head attention with rotational position encoding.
        k: dimension of rotational features
        d: dimension of input features
        num_heads: number of attention heads
        dropout: attention dropout rate
        shared_omega_q: optional shared Omega matrix for Q (or both Q&K if double_omega=False)
        shared_omega_k: optional shared Omega matrix for K (only used if double_omega=True)
        double_omega: if True, use separate Omega matrices for Q and K rotations
        freeze_omega: if True, make Omega matrices non-trainable
        init_omega: initialization strategy for Omega matrix
            - "zero": initialize with zeros
            - "exponential": exponential decay along columns with random frequencies
            - "uniform": standard PyTorch initialization (default)
            - "orthogonal": orthogonal initialization
            - "none": use identity matrix
        attn_type: type of attention
            - "Full": full attention
            - "Linear": linear attention / Performer
        """
        super().__init__()


        self.k = k
        self.d = d
        self.n_head = num_heads
        self.d_head = d // num_heads
        self.init_omega = init_omega
        self.attn_type = attn_type
        self.enable = enable
        self.double_omega = double_omega
        self.freeze_omega = freeze_omega
        self.return_logits = return_logits

        if self.return_logits:
            assert self.attn_type == "Full", "return_logits is only supported for Full attention"

        assert self.d % 2 == 0, "d must be divisible by 2"
        assert self.d_head * num_heads == d, "d must be divisible by num_heads"

        self.WQKV = nn.Linear(d, 3 * d)
        self.WO = nn.Linear(d, d)
        
        if self.attn_type == "Linear":
            assert dropout == 0.0, "dropout is not supported for Performer"
            self.attention = FastAttention(
                dim_heads=self.d_head,
                nb_features=int(self.d_head * math.log(self.d_head))
            )
        else:
            self.dropout = nn.Dropout(dropout)

        if self.enable:
            if shared_omega_q is not None:
                self.OmegaQ = shared_omega_q
            else:
                self.OmegaQ = nn.Linear(k, d//2, bias=False)
                init_omega_matrix(self.OmegaQ, init_omega, d, k)

            if self.double_omega:
                if shared_omega_k is not None:
                    self.OmegaK = shared_omega_k
                else:
                    self.OmegaK = nn.Linear(k, d//2, bias=False)
                    init_omega_matrix(self.OmegaK, init_omega, d, k)
            
            # Freeze omega matrices if requested
            if self.freeze_omega:
                if hasattr(self, 'OmegaQ'):
                    for param in self.OmegaQ.parameters():
                        param.requires_grad = False
                if hasattr(self, 'OmegaK'):
                    for param in self.OmegaK.parameters():
                        param.requires_grad = False
                    
    def forward(self, batch):
        """
        batch.x: (b, n, d) input node features
        batch.t: (b, n, k) (optional) rotational features matrix
        """
        
        x, real_nodes = to_dense_batch(batch.x, batch.batch)

        b, n, _ = x.size()       

        # Linear projections
        QKV = self.WQKV(x)  # (b, n, 3*d)
        Q, K, V = QKV.chunk(3, dim=-1)  # Each (b, n, d)

        # Apply rotation before reshaping for multi-head attention
        if self.enable:
            t, _ = to_dense_batch(batch.t, batch.batch)
            
            phi_q = self.OmegaQ(t)  # (b, n, d//2)
            sin_q = torch.sin(phi_q)  # (b, n, d//2)
            cos_q = torch.cos(phi_q)  # (b, n, d//2)
            
            if self.double_omega:
                phi_k = self.OmegaK(t)  # (b, n, d//2)
                sin_k = torch.sin(phi_k)  # (b, n, d//2)
                cos_k = torch.cos(phi_k)  # (b, n, d//2)
            else:
                sin_k = sin_q
                cos_k = cos_q

            # Apply rotation to Q and K while they are still (b, n, d)
            Q = rotate(Q, sin_q, cos_q)  # (b, n, d)
            K = rotate(K, sin_k, cos_k)  # (b, n, d)

        # Reshape for multi-head attention after rotation
        Q = Q.view(b, n, self.n_head, self.d_head).transpose(1, 2)  # (b, num_heads, n, head_dim)
        K = K.view(b, n, self.n_head, self.d_head).transpose(1, 2)  # (b, num_heads, n, head_dim)
        V = V.view(b, n, self.n_head, self.d_head).transpose(1, 2)  # (b, num_heads, n, head_dim)

        if self.attn_type == "Full":
            # Prefer fused SDPA for lower memory use unless logits are requested
            if not self.return_logits:
                attn_mask = (~real_nodes).unsqueeze(1).unsqueeze(2)  # (b, 1, 1, n), broadcastable to (b, h, n, n)
                context = F.scaled_dot_product_attention(
                    Q, K, V,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False,
                )
            else:
                # Manual path that returns logits; use in-place ops to reduce peak memory
                scores = torch.matmul(Q, K.transpose(-2, -1))  # (b, num_heads, n, n)
                scores.mul_(1 / (self.d_head ** 0.5))
                scores.masked_fill_((~real_nodes).unsqueeze(1).unsqueeze(2), float('-inf'))

                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)

                # Weighted sum of values
                context = torch.matmul(attn_weights, V)

        else:
            # Zero out fake nodes in K and V for linear attention
            mask = real_nodes.unsqueeze(1).unsqueeze(-1)  # (b, 1, n, 1)
            K = K * mask
            V = V * mask
            
            context = self.attention(Q, K, V)

        # Concatenating heads and projecting back
        context = context.transpose(1, 2).contiguous().view(b, n, self.d)
        context = self.WO(context)

        if self.return_logits:
            return context[real_nodes], scores
        else:
            return context[real_nodes]
    
        