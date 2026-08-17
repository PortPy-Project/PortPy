# Copyright 2025, the PortPy Authors
#
# Licensed under the Apache License, Version 2.0 with the Commons Clause restriction.
# ----------------------------------------------------------------------
"""
GPU resampling for the ray-aligned beamlet-dose pipeline (torch.grid_sample).

Replaces the per-beamlet SimpleITK CPU resamples (CT -> ray patch, body -> ray
patch, prediction -> CT/leaf points) with a single ``F.grid_sample`` per call on
a volume that is uploaded to the GPU once and reused across all beamlets. Same
strategy (and numerics) as the DoseRAD2026 ``gpu_resample`` module, generalized
to an arbitrary (origin, spacing, direction) grid so it works for both the
CT->BEV and BEV->points directions.

Geometry convention (matches SimpleITK and ray_geometry):
  A volume has sitk geometry (origin O, spacing S, direction D 3x3, size N=(Nx,Ny,Nz)).
  Its array is (Nz,Ny,Nx). The physical center of voxel index (ix,iy,iz) is
      world = O + D @ (S * (ix,iy,iz)).
  So the continuous index of a world point is
      idx = inv(D) @ (world - O) / S      (= (world-O) @ inv(D).T / S for row points).
  grid_sample(align_corners=True) maps normalized c in [-1,1] to index (c+1)/2*(N-1),
  and expects the last grid axis ordered (x,y,z) = (ix,iy,iz) for an (N,C,Nz,Ny,Nx)
  input. Out-of-volume samples are filled with ``default`` via the
  (subtract default -> grid_sample zeros -> add back) trick, matching sitk's
  DefaultPixelValue.
"""
import numpy as np
import torch
import torch.nn.functional as F


def _t(a, device):
    return torch.as_tensor(np.asarray(a, np.float32), device=device)


def world_to_norm_grid(world_pts, origin, spacing, direction_3x3, size_xyz):
    """
    Map world points (LPS mm) to grid_sample normalized coords for a volume.

    world_pts : (..., 3) torch tensor on the target device.
    origin, spacing : length-3. direction_3x3 : 3x3. size_xyz : (Nx,Ny,Nz).
    Returns (..., 3) normalized to [-1, 1] in (x, y, z) order (align_corners=True).
    """
    device = world_pts.device
    world_pts = world_pts.float()
    O = _t(origin, device)
    S = _t(spacing, device)
    DinvT = _t(np.linalg.inv(np.asarray(direction_3x3, float).reshape(3, 3)).T, device)
    N = _t(size_xyz, device)
    idx = ((world_pts - O) @ DinvT) / S        # continuous (ix, iy, iz)
    return 2.0 * idx / (N - 1.0) - 1.0


def sample_volume_at_world(vol_zyx, world_pts, origin, spacing, direction_3x3,
                           size_xyz, default=0.0, mode="bilinear"):
    """
    Sample a GPU volume at arbitrary world points (trilinear, GPU).

    vol_zyx   : (Nz, Ny, Nx) torch tensor (the volume to sample).
    world_pts : (..., 3) world coords; output keeps the leading shape.
    default   : value for points outside the volume (like sitk DefaultPixelValue).
    mode      : 'bilinear' (linear) or 'nearest'.
    """
    device = vol_zyx.device
    wp = world_pts if torch.is_tensor(world_pts) else _t(world_pts, device)
    wp = wp.to(device)
    lead = wp.shape[:-1]
    grid = world_to_norm_grid(wp.reshape(-1, 3), origin, spacing, direction_3x3, size_xyz)
    grid = grid.view(1, 1, 1, -1, 3)                       # (1,1,1,P,3)
    inp = (vol_zyx - float(default))[None, None]           # (1,1,Nz,Ny,Nx)
    out = F.grid_sample(inp, grid, mode=mode, align_corners=True, padding_mode="zeros")
    return out.view(*lead) + float(default) if lead else out.reshape(())


def ray_voxel_world_centers(origin, spacing, direction_3x3, out_size, device):
    """
    World-coord centers of every voxel in a ray patch, as (Nz, Ny, Nx, 3).

    origin = center of voxel (0,0,0); columns of D are the patch x/y/z axes.
    out_size = (Nx, Ny, Nz) (ray_geometry convention).
    """
    O = _t(origin, device)
    S = _t(spacing, device)
    D = _t(np.asarray(direction_3x3, float).reshape(3, 3), device)
    Nx, Ny, Nz = (int(out_size[0]), int(out_size[1]), int(out_size[2]))
    ix = torch.arange(Nx, device=device, dtype=torch.float32)
    iy = torch.arange(Ny, device=device, dtype=torch.float32)
    iz = torch.arange(Nz, device=device, dtype=torch.float32)
    wx = (ix * S[0]).view(1, 1, Nx, 1) * D[:, 0]          # (1,1,Nx,3)
    wy = (iy * S[1]).view(1, Ny, 1, 1) * D[:, 1]          # (1,Ny,1,3)
    wz = (iz * S[2]).view(Nz, 1, 1, 1) * D[:, 2]          # (Nz,1,1,3)
    return O + wx + wy + wz                                # (Nz,Ny,Nx,3)


def ct_to_ray_patch(ct_zyx_gpu, ct_origin, ct_spacing, ct_direction, ct_size,
                    ray_origin, ray_spacing, ray_direction, ray_out_size,
                    default=-1000.0, mode="bilinear"):
    """Resample a CT (or body) volume onto a ray patch -> (Nz,Ny,Nx) GPU tensor."""
    world = ray_voxel_world_centers(ray_origin, ray_spacing, ray_direction,
                                    ray_out_size, ct_zyx_gpu.device)
    return sample_volume_at_world(ct_zyx_gpu, world, ct_origin, ct_spacing,
                                  ct_direction, ct_size, default=default, mode=mode)