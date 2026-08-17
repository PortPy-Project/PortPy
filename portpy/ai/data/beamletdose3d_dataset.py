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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os.path
from portpy.ai.data.base_dataset import BaseDataset

import numpy as np
import torch

import glob

# BEV samples must be normalized exactly as RayUNetPredictor normalizes its inputs,
# so the channel builders are imported from the inference package rather than
# re-stated here. Both modules are numpy-only at import time.
from portpy.ai.inference.ray_geometry import compute_ray_patch_d1_d2_split
from portpy.ai.inference.inf_matrix_predictor import CT_NORMALIZERS, DEFAULT_NORM
class BeamletDose3DDataset(BaseDataset):
    """Dataset for 3D dose prediction from beamlet features.
            Expects a directory structure:
        root/
          patient_id_1/
            ct.npz
            beams/
              beam_{beam_id}_open.npz
            beamlets/
              bl_beam{beam_id}_col{col_id}.npz
          patient_id_2/
            ...
        Each beamlet file contains:
        - d1: distance to entry
        - d2: distance to exit
        - body_mask: binary mask of patient body
        - target: dose distribution for that beamlet (for training)
        The dataset normalizes CT and beamlet features, and can extract patches if needed.
    """
    def __init__(self, opt, transform=None, patch_size=32, ct_clip=(-1000.0, 3071.0), beams_quantile=0.99, eps=1e-6,
                 patient_id=None,   # NEW: patient_id filter for loading a particular patient during inference
                 bev_channels=('ct', 'd1', 'd1_out'), bev_ct_norm='density', bev_y_max=0.9):
        # Support both:
        # 1) PortPy workflow: opt has dataroot and phase
        # 2) build_apred workflow: opt is directly a root path string
        if isinstance(opt, str):
            self.opt = None
            self.root = opt
        else:
            BaseDataset.__init__(self, opt)
            self.root = os.path.join(opt.dataroot, opt.phase)
        # self.files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pt')])
        self.transform = transform
        self.patch_size = patch_size

        # normalization config
        self.ct_clip = ct_clip
        self.beams_quantile = beams_quantile
        self.eps = eps
        self.patient_id = patient_id
        # BEV config -- must match the RayUNetPredictor arguments used at inference
        # (see examples/python_files/inf_matrix_portpy_beams_voxels_cross_eval.py:
        #  channels=('ct','d1','d1_out'), ct_norm='density', unscale=0.09 = y_max/10).
        self.bev_channels = tuple(bev_channels)
        self.bev_ct_norm = bev_ct_norm
        self.bev_y_max = float(bev_y_max)

        # # collect all beamlet files
        self.samples = []
        # for pid in sorted(os.listdir(self.root)):
        #     pdir = os.path.join(self.root, pid)
        #     if not os.path.isdir(pdir): continue
        #     for f in sorted(glob.glob(os.path.join(pdir, 'beamlets', 'bl_beam*_col*.npz'))):
        #         self.samples.append(f)
        patient_dirs = sorted(os.listdir(self.root))
        if self.patient_id is not None:
            patient_dirs = [p for p in patient_dirs if p == self.patient_id]

        for pid in patient_dirs:
            pdir = os.path.join(self.root, pid)
            if not os.path.isdir(pdir):
                continue
            for f in sorted(glob.glob(os.path.join(pdir, 'beamlets', 'bl_beam*_col*.npz'))):
                self.samples.append(f)

        # simple caches
        self._ct_cache = {}  # pid -> tensor
        self._open_cache = {}  # (pid, beam_id) -> tensor

    def __len__(self):
        return len(self.samples)

    def _get_ct(self, pid):
        if pid not in self._ct_cache:
            obj = np.load(os.path.join(self.root, pid, 'ct.npz'), allow_pickle=True)
            self._ct_cache[pid] = torch.from_numpy(obj['ct']).float()
        return self._ct_cache[pid]

    def _get_open_beam(self, pid, beam_id):
        key = (pid, beam_id)
        if key not in self._open_cache:
            obj = np.load(os.path.join(self.root, pid, 'beams', f'beam_{beam_id}_open.pt'), allow_pickle=True)
            self._open_cache[key] = torch.from_numpy(obj['open_beam_dose']).float()
        return self._open_cache[key]

    def _normalize_ct(self, ct):
        # ct in HU → clip to window then scale to [0,1]
        lo, hi = self.ct_clip
        ct = ct.clamp(min=lo, max=hi)
        ct = (ct - lo) / (hi - lo)
        return ct

    def _normalize_beams(self, beams, mask=None):
        """
        Robust per-sample scaling for beams channel.
        Uses q=0.99 within mask if possible; fallback to global max.
        Then clamp to [0,1].
        """
        # prefer scaling within the mask (where dose is meaningful)
        if mask is not None and mask.sum() > 0:
            vals = beams[mask > 0]
        else:
            vals = beams.view(-1)

        if vals.numel() >= 16:
            denom = torch.quantile(vals, self.beams_quantile)
        else:
            denom = vals.max()

        denom = torch.clamp(denom, min=self.eps)
        beams = beams / denom
        beams = beams.clamp_(0.0, 1.0)
        return beams

    def __getitem__(self, idx):
        bl_path = self.samples[idx]
        item = np.load(bl_path, allow_pickle=True)
        pid = str(item['patient_id'])
        bi = int(item['beam_id'])

        y = torch.from_numpy(item['target'].astype(np.float32))
        mask = y > 0
        body_mask = torch.from_numpy(item['body_mask'].astype(np.float32))

        if 'direction_3x3' in item.files:
            # ---------------- BEV (ray-aligned patch) ----------------
            # Mirrors RayUNetPredictor._build_input exactly: same distance channels
            # from the same function, same normalizers, same channel order. The
            # sample stores only CT/body/target plus the patch geometry, because
            # d1/d2/d1_out are pure functions of that geometry.
            ct_hu = item['ct'].astype(np.float32)
            body_np = item['body_mask'].astype(np.uint8)
            view = tuple(np.asarray(item['view_size_mm'], dtype=np.float64).ravel())
            out = tuple(int(v) for v in np.asarray(item['out_size']).ravel())
            x0_mm = float(item['x0_mm'])
            t_entry = float(item['t_entry'])

            d1, d2, d1_out, _ = compute_ray_patch_d1_d2_split(
                body_np, t_entry if t_entry > 0 else None,
                view_size_mm=view, out_size=out, x0_mm=x0_mm)

            raw = {'ct': ct_hu, 'd1': d1, 'd2': d2, 'd1_out': d1_out,
                   'body': body_np.astype(np.float32)}
            norm = dict(DEFAULT_NORM)
            norm['ct'] = CT_NORMALIZERS[self.bev_ct_norm]
            x_model = torch.from_numpy(
                np.stack([norm[c](raw[c]) for c in self.bev_channels], axis=0).astype(np.float32))

            y_max = self.bev_y_max          # training used y_norm = y / 0.9 * 10
            y_norm = y / y_max * 10
            alpha = 0.1
            hot_mask = (y_norm >= alpha).float() * mask
            scatter_mask = (y_norm < alpha).float() * mask
            x_norm = x_model
        else:
            # ---------------- patient grid (unchanged) ----------------
            ct = self._get_ct(pid)
            # beams = self._get_open_beam(pid, bi)
            d_mm_max = float(item.get('d_mm_max', 2000.0))

            d1 = torch.from_numpy(item['d1'].astype(np.float32)) * (d_mm_max / 65535.0)
            d2 = torch.from_numpy(item['d2'].astype(np.float32)) * (d_mm_max / 65535.0)

            t_entry_mm = float(item.get('t_entry', 0.0))
            d1_out = body_mask * t_entry_mm

            # normalize
            ct_norm = self._normalize_ct(ct)
            # beams_norm = self._normalize_beams(beams, mask)
            d2_norm = d2/600 # global max normalization

            d1_out_norm = d1_out/800
            d1_norm = d1 / 1600  # global max normalization
            # ratio tau with d2/d1_in for hot and scatter ~0.5. If tau star from plots ~0.06. for all mask ~2

            # y_max = y.max()
            y_max = 0.7
            y_norm = y / y_max*10
            alpha = 0.1  # 1% of peak (since already normalized and multiplied by 10)
            hot_mask = (y_norm >= alpha).float() * mask
            scatter_mask = (y_norm < alpha).float() * mask

            x_norm = torch.stack([ct_norm, d2_norm, d1_norm, d1_out_norm, hot_mask, scatter_mask, body_mask], dim=0) #rpl_norm, beams_norm, d1_in_norm,
            x_model = x_norm[0:4]

        # x_patch, y_patch, mask_patch = extract_patch(x, y, mask, patch_size=self.patch_size)
        # return x_patch, y_patch
        if self.transform:
            x_norm = self.transform(x_norm)
        meta = {"patient_id": pid, "beam_id": bi, "col": int(item["col"]),
                "origin_xyz_mm": item["origin_xyz_mm"].astype(np.float32),
                "spacing_xyz_mm": item["spacing_xyz_mm"].astype(np.float32),
                "size_xyz": item["size_xyz"].astype(np.int16)
                }
        # BEV patches are rotated, so the grid orientation has to travel with the sample
        # for the prediction to be placed back on the CT grid correctly.
        if "direction_3x3" in item.files:
            meta["direction_3x3"] = item["direction_3x3"].astype(np.float32)
        return {
            # patient grid: CT, d2, d1, d1_out. BEV: the bev_channels stack (CT, d1, d1_out).
            "A": x_model.float(),
            "B": y_norm.unsqueeze(0).float(),
            "HOT_MASK": hot_mask.unsqueeze(0).float(),
            "SCATTER_MASK": scatter_mask.unsqueeze(0).float(),
            "BODY": body_mask.unsqueeze(0).float(),
            "A_paths": bl_path,
            "B_paths": bl_path,
            "meta": meta,
        }

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

def collate_with_meta(batch):
    return {
        "A": torch.stack([b["A"] for b in batch], dim=0),
        "B": torch.stack([b["B"] for b in batch], dim=0),
        "HOT_MASK": torch.stack([b["HOT_MASK"] for b in batch], dim=0),
        "SCATTER_MASK": torch.stack([b["SCATTER_MASK"] for b in batch], dim=0),
        "BODY": torch.stack([b["BODY"] for b in batch], dim=0),
        "A_paths": [b["A_paths"] for b in batch],
        "B_paths": [b["B_paths"] for b in batch],
        "meta": {
            k: [b["meta"][k] for b in batch]
            for k in batch[0]["meta"].keys()
        },
    }

