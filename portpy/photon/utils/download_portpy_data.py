#!/usr/bin/env python3
"""
download_portpy.py

Download selected PortPy patients (and optionally only selected beams) from Hugging Face Hub.

Examples
--------
# 1) Download 2 patients with planner beams only (plus DicomFiles)
python download_portpy_data.py --patients Lung_Patient_11 Lung_Patient_12 --beam-mode planner --out ./portpy_downloads

# 2) Download all beams for one patient
python download_portpy_data.py --patients Lung_Patient_11 --beam-mode all --out ./portpy_downloads

# 3) Download specific beams for one patient
python download_portpy_data.py --patients Lung_Patient_11 --beam-mode ids --beam-ids 0,3,6,9 --out ./portpy_downloads

Notes
-----
- Public datasets do NOT require a token. If you set HF_TOKEN env var, we pass it as token=...
- snapshot_download() downloads concurrently and caches; reruns are fast and incremental.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union
try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    raise ImportError("To download data please install with: pip install 'portpy[data]'")


REPO_DEFAULT = "PortPy-Project/PortPy_Dataset"

STATIC_FILES = [
    "CT_Data.h5", "CT_MetaData.json",
    "StructureSet_Data.h5", "StructureSet_MetaData.json",
    "OptimizationVoxels_Data.h5", "OptimizationVoxels_MetaData.json",
    "PlannerBeams.json",
]


def _load_planner_beam_ids(repo_id: str, patient_id: str, out_dir: Path, token: Optional[str]) -> List[int]:
    planner_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"data/{patient_id}/PlannerBeams.json",
        local_dir=str(out_dir),
        token=token or None,
    )
    with open(planner_path, "r") as f:
        data = json.load(f)
    ids = data.get("IDs", []) or []
    beam_ids: List[int] = []
    for x in ids:
        beam_ids.append(int(x))
    return beam_ids


def _build_allow_patterns(patient_id: str, beam_mode: str, beam_ids: Optional[List[int]]) -> List[str]:
    patterns: List[str] = []

    for fn in STATIC_FILES:
        patterns.append(f"data/{patient_id}/{fn}")

    # DICOMs (safe if folder doesn't exist)
    patterns.append(f"data/{patient_id}/DicomFiles/**")

    if beam_mode == "none":
        return patterns

    if beam_mode == "all":
        patterns.append(f"data/{patient_id}/Beams/Beam_*_Data.h5")
        patterns.append(f"data/{patient_id}/Beams/Beam_*_MetaData.json")
        return patterns

    # "planner" or "ids"
    if beam_ids:
        for bid in beam_ids:
            patterns.append(f"data/{patient_id}/Beams/Beam_{bid}_Data.h5")
            patterns.append(f"data/{patient_id}/Beams/Beam_{bid}_MetaData.json")
    return patterns


def download_portpy_data(
    patients: Union[str, Sequence[str]],
    *,
    out: Union[str, Path] = "./portpy_downloads",
    beam_mode: str = "planner",          # "planner" | "all" | "ids" | "none"
    beam_ids: Optional[Sequence[int]] = None,
    repo_id: str = REPO_DEFAULT,
    token: Optional[str] = None,
    max_workers: int = 8,
) -> Path:
    """
    Download PortPy patient data to disk (resumable + cached via Hugging Face Hub).

    Parameters
    ----------
    patients : str or list[str]
        Patient id(s), e.g. "Lung_Patient_11" or ["Lung_Patient_11", "Lung_Patient_12"].
    out : str or Path
        Output directory. Data is downloaded under this folder.
    beam_mode : str
        "planner" (default), "all", "ids", or "none".
    beam_ids : list[int] or None
        Used only when beam_mode == "ids".
    repo_id : str
        HF dataset repo id.
    token : str or None
        HF token. Not required for public datasets.
    max_workers : int
        Concurrent download workers.

    Returns
    -------
    Path
        Output directory path.
    """
    if isinstance(patients, str):
        patient_list = [patients]
    else:
        patient_list = list(patients)

    out_dir = Path(out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional: shorter cache path to avoid Windows long-path issues
    os.environ.setdefault("HF_HOME", str(out_dir / ".hf_home"))

    # Normalize beam_ids
    beam_ids_list = list(beam_ids) if beam_ids is not None else None

    for pid in patient_list:
        pid = str(pid).strip()
        if not pid:
            continue

        if beam_mode == "planner":
            resolved_beam_ids = _load_planner_beam_ids(repo_id, pid, out_dir, token)
        elif beam_mode == "ids":
            if not beam_ids_list:
                raise ValueError("beam_mode='ids' requires beam_ids=[...]")
            resolved_beam_ids = [int(x) for x in beam_ids_list]
        else:
            resolved_beam_ids = None

        allow_patterns = _build_allow_patterns(pid, beam_mode, resolved_beam_ids)

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(out_dir),
            allow_patterns=allow_patterns,
            token=token,
            max_workers=max_workers,
        )

    return out_dir

def main():
    import argparse, os

    p = argparse.ArgumentParser()
    p.add_argument("--patients", nargs="+", required=True)
    p.add_argument("--out", default="./portpy_downloads")
    p.add_argument("--beam-mode", choices=["planner", "all", "ids", "none"], default="planner")
    p.add_argument("--beam-ids", default="")
    p.add_argument("--token", default=os.getenv("HF_TOKEN", ""))
    p.add_argument("--max-workers", type=int, default=8)
    args = p.parse_args()

    beam_ids = None
    if args.beam_mode == "ids":
        beam_ids = [int(x) for x in args.beam_ids.split(",") if x.strip()]

    download_portpy_data(
        args.patients,
        out=args.out,
        beam_mode=args.beam_mode,
        beam_ids=beam_ids,
        token=args.token.strip() or None,
        max_workers=args.max_workers,
    )

if __name__ == "__main__":
    main()