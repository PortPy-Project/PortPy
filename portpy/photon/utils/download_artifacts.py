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
"""
download_artifacts.py

Download published PortPy artifacts: pretrained model weights and MIP reference solutions.

These live as release assets on https://github.com/PortPy-Project/artifacts rather than in
the PortPy repository, because they are expensive to regenerate -- the weights are GPU-days
of training and each MIP reference solution is roughly 90 solver-hours.

By default the newest release is used, via GitHub's own /releases/latest/download/ URL, so
publishing a new release (v2.0, say) reaches users with no change to PortPy and no API call.
Pass tag="v1.0.0" to pin a specific vintage instead; a published tag never changes.

One consequence worth knowing: a cached file is matched by NAME, so a copy already fetched
from an older release is not replaced when a newer one appears. Pass force=True to refresh
it. When a future release CHANGES an existing artifact rather than adding one, give it a new
file name -- that is why torchvision puts a hash in its weight file names.

Examples
--------
# 1) List what the default release contains
python download_artifacts.py --list

# 2) Download one artifact
python download_artifacts.py --names mednext_lung.pth --out ./portpy_artifacts

# 3) From Python
import portpy.photon as pp
path = pp.download_artifact("imrt_mip_Lung_Patient_6.npz")

Notes
-----
- Public releases do NOT require a token.
- A cached file is reused, so reruns are fast.
- ARTIFACTS below is documentation, not a gate: any asset name in the release works.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

REPO_DEFAULT = "PortPy-Project/artifacts"

# "latest" resolves through GitHub to the newest non-prerelease release. Any other value is
# treated as an exact tag, e.g. "v1.0.0".
TAG_DEFAULT = "latest"

# The release the table below documents. Downloads follow TAG_DEFAULT, so this only says
# which vintage these descriptions were written against.
RELEASE_DOCUMENTED = "v1.0.0"

# Published artifacts, mapped to (netG, description). netG is the generator name for
# portpy.ai.models.networks3d.define_G, and is None for a reference solution.
ARTIFACTS: Dict[str, Tuple[Optional[str], str]] = {
    # Beamlet dose prediction. Patient grid, 4 input channels.
    "beamlet_unet3d.pth": ("beamlet_unet3d", "Beamlet dose prediction, U-Net3D"),
    "beamlet_vit3d.pth": ("beamlet_vit3d", "Beamlet dose prediction, ViT-3D"),
    "beamlet_swin_unetr3d.pth": ("beamlet_swin_unetr3d",
                                 "Beamlet dose prediction, Swin UNETR-3D"),
    # Total dose prediction. 8 input channels.
    "mednext_lung.pth": ("mednext", "Total dose prediction, MedNeXt, lung"),
    "mednext_prostate.pth": ("mednext", "Total dose prediction, MedNeXt, prostate"),
    "mednext_attn_lung.pth": ("mednext_attn",
                              "Total dose prediction, MedNeXt-Attention, lung"),
    "mednext_attn_prostate.pth": ("mednext_attn",
                                  "Total dose prediction, MedNeXt-Attention, prostate"),
    "mednext_duo_joint.pth": ("mednext_duo",
                              "Total dose prediction, MedNeXt-DualDecoder, both sites, "
                              "build with mednext_n_channels=16"),
    # MIP reference solutions. Each stands in for about 90 solver-hours.
    "imrt_mip_Lung_Patient_6.npz": (None,
                                     "IMRT direct-aperture MIP solution, Lung_Patient_6"),
    "vmat_mip_Lung_Patient_6.npz": (None, "VMAT MIP solution, Lung_Patient_6"),
}

# A download smaller than this is an error page or a truncated transfer, not an artifact.
MIN_BYTES = 1024


def list_artifacts() -> Dict[str, Tuple[Optional[str], str]]:
    """
    Print the known artifacts with a short description of each.

    Returns
    -------
    dict
        The ARTIFACTS table, keyed by asset name.
    """
    print(f"{REPO_DEFAULT}, as published in {RELEASE_DOCUMENTED} "
          f"(downloads follow the {TAG_DEFAULT} release)")
    print("{:<30} {:<22} {}".format("artifact", "netG", "description"))
    print("-" * 96)
    for name, (net_g, description) in ARTIFACTS.items():
        print("{:<30} {:<22} {}".format(name, net_g or "-", description))
    return ARTIFACTS


def download_artifact(
    name: str,
    *,
    out: Union[str, Path] = "./portpy_artifacts",
    tag: str = TAG_DEFAULT,
    repo_id: str = REPO_DEFAULT,
    force: bool = False,
) -> Path:
    """
    Download one published artifact to disk (cached; reruns are fast).

    Parameters
    ----------
    name : str
        Asset file name, e.g. "beamlet_vit3d.pth". Any name present in the release
        works; it does not have to appear in ARTIFACTS.
    out : str or Path
        Output directory. The file is written directly inside it.
    tag : str
        "latest" (default) follows the newest release, so a new release reaches you
        without updating PortPy. Give an exact tag such as "v1.0.0" to pin a vintage.
        Note a cached file is matched by name, so use force=True to refresh one that
        a newer release has changed.
    repo_id : str
        GitHub repository holding the release assets.
    force : bool
        Re-download even when a cached copy is already present.

    Returns
    -------
    Path
        Path to the downloaded file.

    Raises
    ------
    IOError
        If the download fails, or comes back truncated or empty.
    """
    out_dir = Path(out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / name

    if dest.exists() and not force and dest.stat().st_size > MIN_BYTES:
        return dest

    if tag == "latest":
        # GitHub redirects this to the newest non-prerelease release.
        url = f"https://github.com/{repo_id}/releases/latest/download/{name}"
    else:
        url = f"https://github.com/{repo_id}/releases/download/{tag}/{name}"
    tmp = dest.with_name(dest.name + ".part")
    print(f"[portpy] downloading {name} from {repo_id} {tag}")
    try:
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            copied = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                copied += len(chunk)
                if total:
                    size = ("{:.1f} MB".format(total / 2 ** 20) if total >= 2 ** 20
                            else "{:.0f} KB".format(total / 1024))
                    print("\r[portpy]   {:5.1f}% of {}".format(
                        100.0 * copied / total, size), end="")
            if total:
                print()
        # A truncated transfer is the realistic failure mode, so check the length.
        if total and copied != total:
            raise IOError(f"{name} is incomplete: got {copied} of {total} bytes")
        if copied <= MIN_BYTES:
            raise IOError(f"{name} came back empty or too small to be valid")
        tmp.replace(dest)
    except urllib.error.HTTPError as err:
        hint = "" if name in ARTIFACTS else (
            f" '{name}' is not one of the artifacts PortPy knows about; as of "
            f"{RELEASE_DOCUMENTED} those are: {', '.join(sorted(ARTIFACTS))}.")
        raise IOError(f"could not download {name} (HTTP {err.code}).{hint} "
                      f"See https://github.com/{repo_id}/releases") from err
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def download_artifacts(
    names: Optional[Sequence[str]] = None,
    *,
    out: Union[str, Path] = "./portpy_artifacts",
    tag: str = TAG_DEFAULT,
    repo_id: str = REPO_DEFAULT,
    force: bool = False,
) -> Path:
    """
    Download several artifacts, or the whole default release.

    Parameters
    ----------
    names : list[str] or None
        Asset names to fetch. None downloads every entry of ARTIFACTS (about 910 MB).
    out : str or Path
        Output directory.
    tag : str
        "latest" (default) follows the newest release; give an exact tag to pin one.
    repo_id : str
        GitHub repository holding the release assets.
    force : bool
        Re-download even when a cached copy is already present.

    Returns
    -------
    Path
        Output directory path.
    """
    wanted = list(names) if names else list(ARTIFACTS)
    for name in wanted:
        download_artifact(name, out=out, tag=tag, repo_id=repo_id, force=force)
    return Path(out).expanduser().resolve()


def main():
    import argparse

    p = argparse.ArgumentParser(description="Download published PortPy artifacts.")
    p.add_argument("--names", nargs="+", default=None,
                   help="asset names; omit to download the whole release")
    p.add_argument("--out", default="./portpy_artifacts")
    p.add_argument("--tag", default=TAG_DEFAULT,
                   help='"latest" (default) or an exact tag such as v1.0.0')
    p.add_argument("--repo-id", default=REPO_DEFAULT)
    p.add_argument("--force", action="store_true")
    p.add_argument("--list", action="store_true", help="list the artifacts and exit")
    args = p.parse_args()

    if args.list:
        list_artifacts()
        return

    download_artifacts(args.names, out=args.out, tag=args.tag,
                       repo_id=args.repo_id, force=args.force)


if __name__ == "__main__":
    main()
