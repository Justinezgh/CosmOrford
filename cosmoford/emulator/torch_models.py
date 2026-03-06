from typing import Dict, Any

import torch
from torch import nn

from diffusers import UNet2DConditionModel

from typing import Optional, Callable, Any

import torch
import torch.nn as nn

try:
    # Optional, only for better editor support. The function works with nn.Module either way.
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel  # type: ignore
except Exception:  # pragma: no cover - optional import at authoring time
    UNet2DConditionModel = nn.Module  # type: ignore


def patch_unet2dcondition_for_y(
    unet: nn.Module,
    y_dim: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """
    Patch a diffusers.UNet2DConditionModel to accept an additional conditioning vector `y` by
    projecting it into the time embedding space and adding it to the UNet's internal time embedding.

    Notes
    -----
    - This mirrors the Flax variant where `combined_emb = time_emb + y_emb (+ optional aug_emb)`.
    - Implementation uses a forward hook on the module's `time_embedding` to inject `y_emb`.
    - The patched model exposes a new forward signature: forward(..., y: Optional[Tensor] = None, ...).
      When `y` is None, behavior is identical to the original model.

    Parameters
    ----------
    unet: diffusers.UNet2DConditionModel (or compatible)
        The base PyTorch UNet model from diffusers.
    y_dim: int
        Dimension of the additional conditioning vector `y`.
    device: Optional[torch.device]
        Device to place the newly created projection layer on. Defaults to the unet's first parameter device.
    dtype: Optional[torch.dtype]
        Dtype for the projection layer. Defaults to the unet's first parameter dtype.

    Returns
    -------
    nn.Module
        The same instance, patched in-place with:
        - attribute `y_embedding: nn.Linear(y_dim, time_embed_dim)`;
        - a wrapped `forward` that accepts `y` and injects it into the time embedding.

    Example
    -------
    >>> from diffusers import UNet2DConditionModel
    >>> base_unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
    >>> patch_unet2dcondition_for_y(base_unet, y_dim=2)
    >>> x = torch.randn(1, base_unet.config.in_channels, 64, 64)
    >>> t = torch.tensor([10], dtype=torch.long)
    >>> y = torch.zeros(1, 2)
    >>> enc = torch.randn(1, 1, base_unet.config.cross_attention_dim)
    >>> _ = base_unet(x, t, encoder_hidden_states=enc, y=y)
    """

    # Derive time embedding dim from config (same as Flax: 4 * block_out_channels[0])
    time_embed_dim = int(unet.config.block_out_channels[0]) * 4

    # Infer device/dtype from model if not provided
    try:
        p0 = next(unet.parameters())
        default_device = p0.device
        default_dtype = p0.dtype
    except StopIteration:  # uninitialized
        default_device = torch.device("cpu")
        default_dtype = torch.float32

    device = device or default_device
    dtype = dtype or default_dtype

    # Create and register the projection layer on the UNet
    y_embedding = nn.Linear(y_dim, time_embed_dim, bias=True).to(device=device, dtype=dtype)
    setattr(unet, "y_embedding", y_embedding)

    if not hasattr(unet, "time_embedding"):
        raise AttributeError(
            "UNet2DConditionModel missing attribute 'time_embedding'. This patch expects standard diffusers UNet."
        )

    # Preserve the original forward exactly
    if not hasattr(unet, "_original_forward"):
        unet._original_forward = unet.forward  # type: ignore[attr-defined]

    def _forward_with_y(
        sample: torch.Tensor,
        timesteps: torch.Tensor | float | int,
        encoder_hidden_states: torch.Tensor,
        *,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        """Wrapper forward that optionally injects y into the time embedding via a hook."""
        # Fast path: no y provided -> exact original behavior
        if y is None:
            return unet._original_forward(sample, timesteps, encoder_hidden_states, **kwargs)  # type: ignore[attr-defined]

        # Validate/prepare y
        if y.dim() != 2 or y.size(-1) != y_dim:
            raise ValueError(f"Expected y of shape (batch, {y_dim}), got {tuple(y.shape)}")

        # Align y to the y_embedding parameter device/dtype (handles model.to(...) after patching)
        proj_weight = unet.y_embedding.weight  # type: ignore[attr-defined]
        target_device = proj_weight.device
        target_dtype = proj_weight.dtype
        if y.device != target_device or y.dtype != target_dtype:
            y = y.to(device=target_device, dtype=target_dtype)

        # We compute y_emb now and store it on the instance for the hook to pick up (read-only)
        y_emb = unet.y_embedding(y)  # type: ignore[attr-defined]
        # Sanity: batch sizes should match the time embedding's batch during forward
        expected_b = sample.shape[0]
        if y_emb.shape[0] != expected_b:
            raise ValueError(
                f"Batch mismatch between sample ({expected_b}) and y ({y_emb.shape[0]}). Ensure they align."
            )

        # Hook that adds y_emb to the output of time_embedding
        def _inject_y_hook(module: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor):
            # out is (batch, time_embed_dim)
            return out + y_emb

        hook_handle = unet.time_embedding.register_forward_hook(_inject_y_hook)  # type: ignore[attr-defined]
        try:
            return unet._original_forward(sample, timesteps, encoder_hidden_states, **kwargs)  # type: ignore[attr-defined]
        finally:
            hook_handle.remove()

    # Bind wrapper as the new forward (preserving name for nicer traces)
    _forward_with_y.__name__ = "forward"  # type: ignore[attr-defined]
    unet.forward = _forward_with_y  # type: ignore[assignment]

    return unet


def build_unet2d_condition_with_y(config_unet: Dict[str, Any]) -> nn.Module:
    """
    Build a PyTorch diffusers UNet2DConditionModel from a config dict that may include 'y_dim',
    and patch it to accept extra conditioning y. Keys accepted include:
      - in_channels, out_channels, down_block_types, up_block_types,
        block_out_channels, layers_per_block, sample_size, cross_attention_dim (optional), y_dim (required for patch).
    """
    cfg = dict(config_unet)
    y_dim = int(cfg.pop("y_dim"))  # remove from config passed to diffusers

    # If cross-attention is not intended, remove the key to avoid None being indexed internally
    if cfg.get("cross_attention_dim", None) in (None, 0):
        cfg.pop("cross_attention_dim", None)
        # Ensure we don't accidentally use a cross-attention mid block by default
        cfg.setdefault("mid_block_type", "UNetMidBlock2D")

    # diffusers config will ignore unknown keys; ensure only relevant are passed
    unet = UNet2DConditionModel(**cfg)
    unet = patch_unet2dcondition_for_y(unet, y_dim=y_dim)
    return unet
