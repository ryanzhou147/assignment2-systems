import torch
import math

class FlashAttentionFunctionPyTorch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        _, Nq, d = Q.shape
        _, Nk, _ = K.shape
        
        scale = 1.0 / math.sqrt(d)
        
        Bq, Bk = 16, 16
        Tq = math.ceil(Nq / Bq)
        Tk = math.ceil(Nk / Bk)
        
        O = torch.zeros_like(Q)  # (batch, Nq, d)
        L = torch.zeros(Q.shape[0], Nq, device=Q.device, dtype=Q.dtype)  # (batch, Nq)
        
        # Loop over batch
        for b in range(Q.shape[0]):
            Qb = Q[b]  # (Nq, d)
            Kb = K[b]  # (Nk, d)
            Vb = V[b]  # (Nk, d)
            
            for i in range(Tq):
                i_start, i_end = i * Bq, min((i + 1) * Bq, Nq)
                
                Qi = Qb[i_start:i_end]  # (Bq, d)
                Oi = torch.zeros_like(Qi)
                li = torch.zeros(Qi.shape[0], device=Q.device, dtype=Q.dtype)
                mi = torch.full((Qi.shape[0],), float('-inf'), device=Q.device, dtype=Q.dtype)
                
                for j in range(Tk):
                    j_start, j_end = j * Bk, min((j + 1) * Bk, Nk)
                    
                    Kj = Kb[j_start:j_end]  # (Bk, d)
                    Vj = Vb[j_start:j_end]  # (Bk, d)
                    
                    # S_ij = Qi @ K_j^T / sqrt(d)
                    Sij = (Qi @ Kj.T) * scale  # (Bq, Bk)
                    
                    # m_new = max(m_old, rowmax(S_ij))
                    mi_new = torch.maximum(mi, Sij.max(dim=-1).values)
                    
                    # P_ij = exp(S_ij - m_new)
                    Pij = torch.exp(Sij - mi_new.unsqueeze(-1))
                    
                    # alpha = exp(m_old - m_new)
                    alpha = torch.exp(mi - mi_new)
                    
                    # l_new = alpha * l_old + rowsum(P_ij)
                    li = alpha * li + Pij.sum(dim=-1)
                    
                    # O_new = alpha * O_old + P_ij @ V_j
                    Oi = alpha.unsqueeze(-1) * Oi + Pij @ Vj
                    
                    mi = mi_new
                
                # Final normalization
                O[b, i_start:i_end] = Oi / li.unsqueeze(-1)
                L[b, i_start:i_end] = mi + torch.log(li)
        
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError