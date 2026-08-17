# tests/test_ai_module.py
"""
Fast, data-free unit tests for the portpy.ai module.

Two tiers:
  1. Pure-numpy beam geometry (portpy.ai.inference.ray_geometry) -- always runs.
  2. Torch-dependent tests (model forward pass, GPU/CPU resampling) -- skipped
     automatically when torch is not installed (CI installs CPU torch).

No patient data, no network, no GPU required. Everything runs on CPU in seconds.
"""
# NOTE (Windows): torch must be imported BEFORE portpy/SimpleITK, otherwise its
# DLLs fail to load (OSError WinError 127). Harmless no-op where torch is absent.
try:
    import torch  # noqa: F401
except ImportError:
    pass

import numpy as np
import pytest

from portpy.ai.inference.ray_geometry import (
    get_rotation_matrix,
    source_lps_from_beam,
    beamlet_center_lps,
    build_ray_grid_with_beam_axes,
)


# ---------------------------------------------------------------------------
# 1) Beam geometry (numpy only)
# ---------------------------------------------------------------------------

ANGLES = [(0, 0, 0), (90, 0, 0), (180, 0, 0), (270, 0, 0),
          (160, 0, 90), (120, 0, 90), (45, 10, 30), (300, 350, 270)]


@pytest.mark.parametrize("gantry,couch,coll", ANGLES)
def test_rotation_matrix_is_orthonormal(gantry, couch, coll):
    R = get_rotation_matrix(gantry, couch, coll)
    assert R.shape == (3, 3)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)  # proper rotation


def test_zero_angles_is_identity():
    np.testing.assert_allclose(get_rotation_matrix(0.0, 0.0, 0.0), np.eye(3), atol=1e-12)


def test_collimator_zero_matches_two_arg_call():
    """collimator_deg=0 must be a strict no-op (same as omitting it)."""
    for gantry, couch, _ in ANGLES:
        np.testing.assert_allclose(get_rotation_matrix(gantry, couch),
                                   get_rotation_matrix(gantry, couch, 0.0), atol=1e-15)


@pytest.mark.parametrize("gantry,couch,coll", ANGLES)
def test_collimator_rotates_about_beam_axis(gantry, couch, coll):
    """
    Collimator physically rotates the field about the BEAM CENTRAL AXIS
    (beam-frame y). Consequently the source position (0,-SAD,0 in beam frame,
    ON the rotation axis) must NOT move when the collimator rotates.
    """
    iso = np.array([12.3, -45.6, 78.9])
    SAD = 1000.0
    R0 = get_rotation_matrix(gantry, couch, 0.0)
    Rc = get_rotation_matrix(gantry, couch, coll)
    src0 = source_lps_from_beam(iso, R0, SAD)
    srcc = source_lps_from_beam(iso, Rc, SAD)
    np.testing.assert_allclose(src0, srcc, atol=1e-9)
    # and the central axis (source -> iso) is unchanged too
    np.testing.assert_allclose(iso - src0, iso - srcc, atol=1e-9)


def test_collimator_90_swaps_beamlet_axes():
    """At collimator 90 deg a beamlet at (bx, by) maps to where (-by, bx) sits
    at collimator 0 (right-handed rotation about beam-frame y)."""
    iso = np.zeros(3)
    R0 = get_rotation_matrix(30.0, 0.0, 0.0)
    R90 = get_rotation_matrix(30.0, 0.0, 90.0)
    bx, by = 25.0, 40.0
    p90 = beamlet_center_lps(bx, by, R90, iso)
    p0 = beamlet_center_lps(-by, bx, R0, iso)
    np.testing.assert_allclose(p90, p0, atol=1e-9)


def test_source_at_sad_distance():
    iso = np.array([1.0, 2.0, 3.0])
    for gantry, couch, coll in ANGLES:
        R = get_rotation_matrix(gantry, couch, coll)
        src = source_lps_from_beam(iso, R, 1000.0)
        assert np.isclose(np.linalg.norm(src - iso), 1000.0, atol=1e-9)


def test_source_gantry0_is_anterior():
    """Gantry 0 / couch 0: R = I, so the source sits at iso + (0, -SAD, 0)."""
    iso = np.array([10.0, 20.0, 30.0])
    src = source_lps_from_beam(iso, get_rotation_matrix(0.0, 0.0, 0.0), 1000.0)
    np.testing.assert_allclose(src, iso + [0.0, -1000.0, 0.0], atol=1e-12)


def test_beamlet_center_roundtrip():
    """Mapping the LPS beamlet center back to the beam frame recovers (bx, 0, by)."""
    iso = np.array([-5.0, 15.0, 40.0])
    for gantry, couch, coll in ANGLES:
        R = get_rotation_matrix(gantry, couch, coll)
        bx, by = 17.5, -22.5
        p = beamlet_center_lps(bx, by, R, iso)
        rb = (p - iso) @ R          # row-vector convention: r_beam = r_lps @ R
        np.testing.assert_allclose(rb, [bx, 0.0, by], atol=1e-9)


def test_ray_grid_axes():
    """Ray-patch grid: depth axis points from source toward the beamlet."""
    iso = np.zeros(3)
    R = get_rotation_matrix(160.0, 0.0, 90.0)
    src = source_lps_from_beam(iso, R, 1000.0)
    ctr = beamlet_center_lps(10.0, -5.0, R, iso)
    origin, spacing, D = build_ray_grid_with_beam_axes(
        src, ctr, R, view_size_mm=(512.0, 64.0, 64.0), out_size=(256, 32, 32))
    D = np.asarray(D, float).reshape(3, 3)
    np.testing.assert_allclose(D.T @ D, np.eye(3), atol=1e-9)   # orthonormal axes
    ray_dir = (ctr - src) / np.linalg.norm(ctr - src)
    # first grid axis (depth) is the source->beamlet ray
    np.testing.assert_allclose(D[:, 0], ray_dir, atol=1e-9)
    assert len(np.asarray(spacing)) == 3 and np.all(np.asarray(spacing, float) > 0)


# ---------------------------------------------------------------------------
# 2) Torch-dependent (model + resampling); skipped when torch is unavailable
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch not installed (pip install portpy[ai])")


def test_unet3d_forward():
    """Tiny random-weight forward pass: shape and finiteness."""
    from portpy.ai.models.unet3d import UNet3D

    model = UNet3D(in_channels=3, out_channels=1, base_features=8, num_groups=4)
    model.eval()
    x = torch.randn(1, 3, 16, 16, 16)   # divisible by 8 (three poolings)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 16, 16, 16)
    assert torch.isfinite(y).all()


def test_sample_volume_at_world_trilinear_exact():
    """
    Trilinear interpolation of a LINEAR function is exact: sampling a ramp
    volume f = ix + 2*iy + 3*iz at arbitrary interior world points must
    reproduce the analytic value (this pins the align_corners=True voxel-center
    convention that matches SimpleITK).
    """
    from portpy.ai.inference.gpu_resample import sample_volume_at_world

    nx, ny, nz = 12, 10, 8
    origin = np.array([-5.0, 3.0, 10.0])
    spacing = np.array([2.0, 2.5, 3.0])
    direction = np.eye(3).ravel()

    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    vol_xyz = ix + 2.0 * iy + 3.0 * iz
    vol_zyx = torch.as_tensor(np.transpose(vol_xyz, (2, 1, 0)), dtype=torch.float32)

    rng = np.random.default_rng(0)
    idx = rng.uniform([0.5, 0.5, 0.5], [nx - 1.5, ny - 1.5, nz - 1.5], size=(200, 3))
    world = origin + idx * spacing
    expected = idx[:, 0] + 2.0 * idx[:, 1] + 3.0 * idx[:, 2]

    got = sample_volume_at_world(vol_zyx, torch.as_tensor(world, dtype=torch.float32),
                                 origin, spacing, direction, (nx, ny, nz)).cpu().numpy()
    np.testing.assert_allclose(got, expected, atol=1e-3)


def test_sample_volume_default_outside():
    """Points far outside the volume return the 'default' value."""
    from portpy.ai.inference.gpu_resample import sample_volume_at_world

    vol = torch.ones(4, 4, 4)
    far = torch.tensor([[1e4, 1e4, 1e4]], dtype=torch.float32)
    got = sample_volume_at_world(vol, far, [0, 0, 0], [1, 1, 1],
                                 np.eye(3).ravel(), (4, 4, 4), default=-7.0)
    np.testing.assert_allclose(got.cpu().numpy(), [-7.0], atol=1e-5)
