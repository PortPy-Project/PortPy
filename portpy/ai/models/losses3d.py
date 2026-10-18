# Copyright 2025, the PortPy Authors
#
# Licensed under the Apache License, Version 2.0 with the Commons Clause restriction.
# You may obtain a copy of the Apache 2 License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ----------------------------------------------------------------------
# Commons Clause Restriction Notice:
# PortPy is licensed under Apache 2.0 with the Commons Clause.
# You may use, modify, and share the code for non-commercial
# academic and research purposes only.
# Commercial use — including offering PortPy as a service,
# or incorporating it into a commercial product — requires
# a separate commercial license.
# ----------------------------------------------------------------------

"""Loss functions for 3D dose prediction.

Moved out of ``networks3d.py`` verbatim; behaviour is unchanged. ``networks3d``
re-exports every name here, so ``networks.MomentLoss`` and friends keep working.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DVHLoss(nn.Module):
    def __init__(self):
        super(DVHLoss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def __call__(self, predicted_dose, target_hist, target_bins, oar):
        """
        Calculate DVH loss: averaged over all OARs. Target hist is already computed
            predicted dose (tensor) -- [N, C, D, H, W] C = 1
            target hist (tensor)    -- [N, n_bins, n_oars]
            target bins (tensor)    -- [N, n_bins]
            oar (tensor)            -- [N, C, D, H, W] C == n_oars one hot encoded OAR including PTV
        """

        # Calculate predicted hist
        vols = torch.sum(oar, axis=(2, 3, 4))
        n_bins = target_bins.shape[1]
        hist = torch.zeros_like(target_hist)
        bin_w = target_bins[0, 1] - target_bins[0, 0]

        # print(vols.shape, hist.shape)

        for i in range(n_bins):
            diff = torch.sigmoid((predicted_dose - target_bins[:, i]) / bin_w)
            # print(diff.shape)
            # diff = torch.cat(oar.shape[1] * [diff.unsqueeze(axis=1)]) * oar
            diff = diff.repeat(1, oar.shape[1], 1, 1, 1) * oar
            num = torch.sum(diff, axis=(2, 3, 4))
            # print(diff.shape, num.shape)
            hist[:, i] = (num / vols)

        # print(hist.detach().cpu().numpy())
        # print(vols.detach().cpu().numpy())

        return self.loss(hist, target_hist)


# Defines Bhattacharya loss class
class BhattLoss(nn.Module):
    def __init__(self):
        super(BhattLoss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def __call__(self, predicted_dose, target_hist, target_bins, oar):
        """
        Calculate DVH loss: averaged over all OARs. Target hist is already computed
            predicted dose (tensor) -- [N, C, D, H, W] C = 1
            target hist (tensor)    -- [N, n_bins, n_oars]
            target bins (tensor)    -- [N, n_bins]
            oar (tensor)            -- [N, C, D, H, W] C == n_oars one hot encoded OAR including PTV
        """

        # Calculate predicted hist
        vols = torch.sum(oar, axis=(2, 3, 4))
        n_bins = target_bins.shape[1]
        hist = torch.zeros_like(target_hist)
        bin_w = target_bins[0, 1] - target_bins[0, 0]

        # print(vols.shape, hist.shape)

        for i in range(n_bins):
            diff = torch.sigmoid((predicted_dose - target_bins[:, i]) / bin_w)
            # print(diff.shape)
            # diff = torch.cat(oar.shape[1] * [diff.unsqueeze(axis=1)]) * oar
            diff = diff.repeat(1, oar.shape[1], 1, 1, 1) * oar
            num = torch.sum(diff, axis=(2, 3, 4))
            # print(diff.shape, num.shape)
            hist[:, i] = (num / vols)

        print(hist.detach().cpu().numpy())
        print(vols.detach().cpu().numpy())
        histprod = torch.sqrt(hist * target_hist)
        # histprod = torch.clamp(histprod, 1e-8)
        print(histprod.detach().cpu().numpy())
        bhattdist = torch.sum(histprod, axis=(1, 2))  # Sum of bhattacharya distances for each OAR
        bhattdist = torch.clamp(bhattdist, 1e-3)
        print(bhattdist.detach().cpu().numpy())
        bhattloss = torch.mean(-torch.log(bhattdist))
        return bhattloss


import torch.nn.functional as F

class WeightedL1Loss(nn.Module):
    def __init__(self, alpha=2.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, target):
        denom = torch.clamp(target.amax(dim=(2, 3, 4), keepdim=True), min=self.eps)
        rel = target / denom
        w = 1.0 + self.alpha * rel
        loss = torch.abs(pred - target) * w
        return loss.sum() / torch.clamp(w.sum(), min=self.eps)


class MaskedWeightedL1Loss(nn.Module):
    def __init__(self, alpha=2.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, target, mask):
        denom = torch.clamp(target.amax(dim=(2, 3, 4), keepdim=True), min=self.eps)
        rel = target / denom
        w = (1.0 + self.alpha * rel) * mask
        loss = torch.abs(pred - target) * w
        return loss.sum() / torch.clamp(w.sum(), min=self.eps)

class MomentLoss(nn.Module):
    """
    Matches specified dose moments inside each ROI channel.

    Expected:
      y_pred:    [B, 1, D, H, W]
      y_true:    [B, 1, D, H, W]
      mask_dict: dict[str, Tensor], each tensor shaped [B, C, D, H, W]
                 Example:
                   {
                     "OARPTV": structures_tensor,   # channels = OARs + PTV
                   }

    Computes moments per patient, per channel, then averages.
    """
    def __init__(self, moments=(1, 2, 10), reduction="mean", eps=1e-6):
        super().__init__()
        self.moments = moments
        self.reduction = reduction
        self.eps = eps
        self.loss = nn.MSELoss()

    def _masked_moment(self, dose: torch.Tensor, mask: torch.Tensor, n: int) -> torch.Tensor:
        """
        dose: [B,1,D,H,W]
        mask: [B,1,D,H,W]
        returns: [B] nth raw moment inside mask for each sample
        """
        mask = (mask > 0.5).to(dose.dtype)
        vol = mask.sum(dim=(1, 2, 3, 4)).clamp(min=self.eps)   # [B]
        moment_n = torch.pow(
            ((dose.pow(n) * mask).sum(dim=(1, 2, 3, 4)) / vol).clamp(min=self.eps),
            1.0 / n
        )
        return moment_n

    def forward(self,
                y_pred: torch.Tensor,
                y_true: torch.Tensor,
                mask_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        losses = []

        for n in self.moments:
            for m in mask_dict.values():
                # m: [B,C,D,H,W] or [B,1,D,H,W]
                for c in range(m.size(1)):
                    mask_c = m[:, c:c+1, ...]  # [B,1,D,H,W]

                    # part for skipping empty masks; if vol=0, both moments become 0 and loss becomes 0, so they are automatically skipped in the final averaging
                    mask_bin = (mask_c > 0.5).to(y_true.dtype)
                    vol = mask_bin.sum(dim=(1, 2, 3, 4))  # [B]

                    valid = vol > 0
                    if not valid.any():
                        continue

                    pred_m = self._masked_moment(y_pred, mask_c, n)  # [B]
                    true_m = self._masked_moment(y_true, mask_c, n)  # [B]

                    # skip empty-mask samples automatically because both moments become 0
                    losses.append(torch.abs(pred_m - true_m).mean())
                    # losses.append(self.loss(pred_m[valid], true_m[valid])) # use valid to skip empty-mask samples in average

        if len(losses) == 0:
            return y_true.new_tensor(0.0)

        loss = torch.stack(losses).mean()
        if self.reduction == "mean":
            return loss
        return loss * y_true.numel()


def _central_diff(x: torch.Tensor, dim: int):
    """
    Central finite difference along spatial dimension dim:
      dim = 0 -> depth
      dim = 1 -> height
      dim = 2 -> width

    x shape: [B, 1, D, H, W]
    output shape: same as x
    """
    # F.pad for 5D tensors uses:
    # (W_left, W_right, H_left, H_right, D_left, D_right)
    pad = [0, 0, 0, 0, 0, 0]

    if dim == 0:      # depth
        pad[4] = 1
        pad[5] = 1
    elif dim == 1:    # height
        pad[2] = 1
        pad[3] = 1
    elif dim == 2:    # width
        pad[0] = 1
        pad[1] = 1
    else:
        raise ValueError("dim must be 0, 1, or 2")

    x_pad = F.pad(x, pad, mode="replicate")

    # spatial dims of x are (2, 3, 4)
    spatial_axis = dim + 2

    forward = x_pad.narrow(spatial_axis, 2, x.size(spatial_axis))
    backward = x_pad.narrow(spatial_axis, 0, x.size(spatial_axis))
    return (forward - backward) / 2.0


def _gradient_3d(x: torch.Tensor):
    """
    Returns stacked spatial gradients:
    shape = [3, B, 1, D, H, W]
    """
    return torch.stack([_central_diff(x, d) for d in range(3)], dim=0)


class DualGradientL2Loss(nn.Module):
    """
    Dual gradient loss:
      - edge term emphasizes regions with large GT gradients
      - flat term matches gradients everywhere more uniformly

    Total = lambda_edge * L_edge + lambda_flat * L_flat
    """
    def __init__(self,
                 gamma_edge: int = 25,
                 gamma_flat: int = 0,
                 lambda_edge: float = 0.1,
                 lambda_flat: float = 0.05,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma_edge = gamma_edge
        self.gamma_flat = gamma_flat
        self.lambda_edge = lambda_edge
        self.lambda_flat = lambda_flat
        self.reduction = reduction

    @staticmethod
    def _sharp_term(gp: torch.Tensor, gg: torch.Tensor, gamma: int):
        """
        gp, gg shape: [3, B, 1, D, H, W]
        """
        diff2 = (gp - gg).pow(2).sum(0)   # [B,1,D,H,W]
        weight = (gg.pow(2).sum(0).sqrt() + 1e-6).pow(gamma)
        weight = weight.clamp(max=6)
        return diff2 * weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        gp = _gradient_3d(y_pred)
        gg = _gradient_3d(y_true)

        l_edge = self._sharp_term(gp, gg, self.gamma_edge)
        l_flat = self._sharp_term(gp, gg, self.gamma_flat)

        loss = self.lambda_edge * l_edge + self.lambda_flat * l_flat

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")

class BerHuLoss(nn.Module):
    def __init__(self, delta=1.0, eps=1e-8):
        super().__init__()
        self.delta = delta
        self.eps = eps

    def forward(self, pred, true, mask):
        e = (pred - true) * mask
        ae = e.abs()
        loss = torch.where(
            ae <= self.delta,
            ae,
            (e * e + self.delta * self.delta) / (2 * self.delta)
        )
        return (loss * mask).sum() / (mask.sum() + self.eps)
