import torch.nn as nn
import torch
import numpy as np
from torchdyn.core import NeuralODE

@torch.no_grad()
def solve_ode_forward(
    x0_in, model: nn.Module, theta_in, device: torch.device, rtol: float = 1e-2, atol: float = 1e-2, nb_steps: int = 11
):
    """Integrate dx/dt = f_theta(t, x) and return trajectory (T, B, H, W).

    Accepts x0_in and theta_in as either NumPy arrays or torch.Tensors.
    Handles existing channel dimension to avoid redundant conversions.
    """
    model.eval()
    # Prepare initial field x0
    if isinstance(x0_in, torch.Tensor):
        x0 = x0_in.float().to(device)
        # Accept (B,H,W) or (B,1,H,W); add channel dim only if missing
        if x0.dim() == 3:            # (B,H,W)
            x0 = x0.unsqueeze(1)     # -> (B,1,H,W)
        elif x0.dim() == 4 and x0.shape[1] == 1:  # already (B,1,H,W)
            pass
        else:
            raise ValueError(f"solve_ode_forward expected x0 tensor with shape (B,H,W) or (B,1,H,W), got {tuple(x0.shape)}")
    else:  # NumPy path
        x0_np = np.asarray(x0_in)
        if x0_np.ndim == 3:          # (B,H,W)
            x0 = torch.from_numpy(x0_np).float().unsqueeze(1).to(device)
        elif x0_np.ndim == 4 and x0_np.shape[1] == 1:  # (B,1,H,W)
            x0 = torch.from_numpy(x0_np).float().to(device)
        else:
            raise ValueError(f"solve_ode_forward expected x0 array with shape (B,H,W) or (B,1,H,W), got {x0_np.shape}")

    # Prepare conditioning theta
    if isinstance(theta_in, torch.Tensor):
        y = theta_in.float().to(device)
    else:
        y = torch.from_numpy(np.asarray(theta_in)).float().to(device)

    ts = torch.linspace(0.0, 1.0, steps=nb_steps, device=device)

    class VectorField(nn.Module):
        def __init__(self, net: nn.Module, y: torch.Tensor):
            super().__init__()
            self.net = net
            self.register_buffer('y', y)

        def forward(self, t: torch.Tensor, x: torch.Tensor, args=None, **kwargs) -> torch.Tensor:
            # t may be scalar tensor; broadcast to batch
            if t.dim() == 0:
                t_in = t.repeat(x.size(0))
            else:
                t_in = t
            # Provide encoder_hidden_states only if the UNet expects cross-attention
            enc = None
            cross_dim = getattr(self.net.config, "cross_attention_dim", None)
            if cross_dim is not None:
                enc = torch.zeros(x.size(0), 1, cross_dim, device=x.device, dtype=x.dtype)
            out = self.net(x, t_in, encoder_hidden_states=enc, y=self.y)
            return out.sample if hasattr(out, 'sample') else out

    vf = VectorField(model, y)
    ode = NeuralODE(vf, solver='dopri5', rtol=rtol, atol=atol)
    ys = ode.trajectory(x0, ts)
    # torchdyn may return a tuple or tensor depending on version
    if isinstance(ys, (list, tuple)):
        y_traj = ys[0]
    else:
        y_traj = ys
    # y_traj: (T, B, C, H, W)
    return y_traj.detach().cpu().squeeze(2).numpy()  # (T,B,H,W)
