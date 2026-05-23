import torch
from torch import Tensor
from scipy.optimize import linear_sum_assignment


def minibatch_ot_permute(x_0: Tensor, x_1: Tensor) -> tuple[Tensor, Tensor]:
    """Reorder x_0 and x_1 within a minibatch to minimise squared-Euclidean
    transport cost — the discrete Kantorovich (assignment) problem solved
    exactly via the Hungarian algorithm.

    Shapes: both inputs (B, *), returns (x_0_perm, x_1) where x_0 rows are
    permuted so that paired (x_0_perm[i], x_1[i]) have low joint cost.

    We keep x_1 fixed and permute x_0 for consistency with the training loop
    (x_1 carries the class label that must stay aligned with the conditioning).
    """
    B = x_0.shape[0]
    with torch.no_grad():
        x0_flat = x_0.reshape(B, -1).float()
        x1_flat = x_1.reshape(B, -1).float()
        # Cost matrix: C[i,j] = ||x1[i] - x0[j]||^2
        cost = torch.cdist(x1_flat, x0_flat, p=2).pow(2).cpu().numpy()
        _, col_idx = linear_sum_assignment(cost)
    perm = torch.as_tensor(col_idx, device=x_0.device, dtype=torch.long)
    return x_0[perm], x_1
