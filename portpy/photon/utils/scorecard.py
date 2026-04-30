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

import copy
import json
import numpy as np
import pandas as pd


def load_scorecard_json(scorecard):
    if isinstance(scorecard, str):
        with open(scorecard, "r") as f:
            return json.load(f)
    return copy.deepcopy(scorecard)


def adapt_scorecard_prescription(scorecard_json, old_pres_gy, new_pres_gy):
    """
    Scale a scorecard defined for one prescription to another prescription.
    Only scale values that are explicitly in Gy.
    """
    sc = copy.deepcopy(scorecard_json)
    scale = float(new_pres_gy) / float(old_pres_gy)

    if "DosePerFraction" in sc and "NumberOfFractions" in sc:
        sc["DosePerFraction"] = float(new_pres_gy) / float(sc["NumberOfFractions"])

    for item in sc.get("ScoreTemplates", []):
        metric_type = item.get("MetricType", "")
        input_unit = item.get("InputUnit", None)
        output_unit = item.get("OutputUnit", None)

        if input_unit == "Gy":
            item["InputValue"] = float(item["InputValue"]) * scale

        if output_unit == "Gy":
            for pt in item.get("ScorePoints", []):
                pt["PointX"] = float(pt["PointX"]) * scale

        # ConformationNumber uses InputValue as RI dose threshold in Gy,
        # but score points themselves are unitless.
        if metric_type == "ConformationNumber" and input_unit == "Gy":
            item["InputValue"] = float(item["InputValue"]) * scale

    return sc

def resolve_actual_structure_name(plan, structure_name, alias_map=None):
    if structure_name is None:
        return None

    alias_map = alias_map or {}
    names = plan.structures.get_structures()
    names_upper = [s.upper() for s in names]

    req = structure_name.upper()
    req = alias_map.get(req, req)

    if req in names_upper:
        return names[names_upper.index(req)]

    return None

def get_structure_mask(structs, structure_name, alias_map=None):
    """
    Return boolean mask for a structure, with a few useful aliases/fallbacks.
    """
    alias_map = alias_map or {}
    names = structs.structures_dict["name"]
    names_upper = [s.upper() for s in names]
    masks = structs.structures_dict["structure_mask_3d"]

    req = structure_name.upper()
    req = alias_map.get(req, req)

    if req in names_upper:
        return masks[names_upper.index(req)] > 0

    # Lung scorecard fallback
    if req == "TOTAL LUNG - GTV":
        if "LUNGS_NOT_GTV" in names_upper:
            return masks[names_upper.index("LUNGS_NOT_GTV")] > 0

        lung_union = np.zeros_like(masks[0], dtype=bool)
        if "LUNG_L" in names_upper:
            lung_union |= (masks[names_upper.index("LUNG_L")] > 0)
        if "LUNG_R" in names_upper:
            lung_union |= (masks[names_upper.index("LUNG_R")] > 0)

        if not lung_union.any():
            return None

        if "GTV" in names_upper:
            gtv = masks[names_upper.index("GTV")] > 0
            lung_union &= (~gtv)

        return lung_union

    # Prostate scorecard fallback
    if req == "BILAT_FEM_HEADS":
        req = "FEMURS"

    if req == "FEMURS":
        if "FEMURS" in names_upper:
            return masks[names_upper.index("FEMURS")] > 0

        fem_union = np.zeros_like(masks[0], dtype=bool)
        if "FEMUR_L" in names_upper:
            fem_union |= (masks[names_upper.index("FEMUR_L")] > 0)
        if "FEMUR_R" in names_upper:
            fem_union |= (masks[names_upper.index("FEMUR_R")] > 0)

        return fem_union if fem_union.any() else None

    return None

def add_temp_structure_if_needed(plan, structure_name, alias_map=None):
    if structure_name is None:
        return

    structs = plan.structures
    existing = [s.upper() for s in structs.structures_dict["name"]]
    req = structure_name.upper()

    if req in existing:
        return

    mask = get_structure_mask(structs, req, alias_map=alias_map)
    if mask is None:
        return

    new_struct_name = req
    try:
        # PortPy supports creating optimization structures by explicit masks
        structs.create_structure(
            structure_name=new_struct_name,
            mask_3d=mask.astype(np.uint8)
        )
    except Exception:
        # fallback: do nothing if your local PortPy version doesn't expose this helper
        pass


def interpolate_score(value, score_points):
    """
    Piecewise-linear interpolation through score card points.
    """
    if np.isnan(value):
        return np.nan

    pts = sorted(score_points, key=lambda x: float(x["PointX"]))
    xs = np.array([float(p["PointX"]) for p in pts], dtype=float)
    ys = np.array([float(p["Score"]) for p in pts], dtype=float)

    return float(np.interp(value, xs, ys, left=ys[0], right=ys[-1]))


def _dose_at_volume_cc_with_portpy(plan, dose_1d, struct_name, volume_cc):
    from portpy.photon.evaluation import Evaluation
    """
    Convert cc -> percent and use PortPy Evaluation.get_dose(...)
    """
    vol_cc = plan.structures.get_volume_cc(structure_name=struct_name)
    if vol_cc <= 0:
        return np.nan
    volume_per = 100.0 * float(volume_cc) / float(vol_cc)
    dummy_sol = {"inf_matrix": plan.inf_matrix, "dose_1d": dose_1d}
    return float(Evaluation.get_dose(dummy_sol, dose_1d=dose_1d, struct=struct_name, volume_per=volume_per))


def _dose_at_subvolume_percent_rx(plan, dose_1d, struct_name, subvol_cc, prescription_gy):
    d_gy = _dose_at_volume_cc_with_portpy(plan, dose_1d, struct_name, subvol_cc)
    if np.isnan(d_gy):
        return np.nan
    return 100.0 * d_gy / float(prescription_gy)


def _conformation_number(plan, dose_1d, struct_name, ri_dose_gy):
    """
    CN = (TV_RI^2)/(TV * V_RI)
    Uses the dose grid voxels directly.
    """
    mask = get_structure_mask(plan.structs, struct_name)
    if mask is None:
        return np.nan

    dose_3d = plan.inf_matrix.dose_1d_to_3d(dose_1d=dose_1d)
    ri_mask = dose_3d >= float(ri_dose_gy)

    tv = np.sum(mask > 0)
    tv_ri = np.sum((mask > 0) & ri_mask)
    v_ri = np.sum(ri_mask)

    if tv == 0 or v_ri == 0:
        return np.nan

    return float((tv_ri ** 2) / (tv * v_ri))


def evaluate_score_item(plan, dose_1d, item, prescription_gy=None, alias_map=None):
    from portpy.photon.evaluation import Evaluation
    structure_name = (
            item["Structure"].get("StructureId")
            or item["Structure"].get("TemplateStructureId")
    )
    metric_type = item["MetricType"]
    input_value = float(item.get("InputValue", 0.0))
    input_unit = item.get("InputUnit", None)

    # make sure derived structures are available if possible
    add_temp_structure_if_needed(plan, structure_name, alias_map=alias_map)
    actual_name = resolve_actual_structure_name(plan, structure_name, alias_map=alias_map)
    if actual_name is None:
        return {
            "Structure": structure_name,
            "MetricType": metric_type,
            "MetricValue": np.nan,
            "Score": np.nan,
            "Status": "missing_structure",
            "MetricComment": item.get("MetricComment", None),
        }

    struct_name_for_eval = actual_name
    # struct_name_for_eval = structure_name
    if structure_name.upper() not in [s.upper() for s in plan.structures.get_structures()]:
        # try alias resolution
        alias_map = alias_map or {}
        mapped = alias_map.get(structure_name.upper(), structure_name.upper())
        if mapped in [s.upper() for s in plan.structures.get_structures()]:
            struct_name_for_eval = mapped
        else:
            # if still unavailable, return missing
            return {
                "Structure": structure_name,
                "MetricType": metric_type,
                "MetricValue": np.nan,
                "Score": np.nan,
                "Status": "missing_structure",
                "MetricComment": item.get("MetricComment", None),
            }

    dummy_sol = {"inf_matrix": plan.inf_matrix, "dose_1d": dose_1d}

    try:
        if metric_type == "VolumeAtDose":
            # Output is percent volume
            metric_value = float(
                Evaluation.get_volume(
                    dummy_sol, dose_1d=dose_1d, struct=struct_name_for_eval, dose_value_gy=input_value
                )
            )

        elif metric_type == "DoseAtVolume":
            if input_unit == "%":
                metric_value = float(
                    Evaluation.get_dose(
                        dummy_sol, dose_1d=dose_1d, struct=struct_name_for_eval, volume_per=input_value
                    )
                )
            elif input_unit == "CC":
                metric_value = _dose_at_volume_cc_with_portpy(plan, dose_1d, struct_name_for_eval, input_value)
            else:
                raise ValueError(f"Unsupported InputUnit for DoseAtVolume: {input_unit}")

        elif metric_type == "MeanDose":
            metric_value = float(
                Evaluation.get_mean_dose(dummy_sol, dose_1d=dose_1d, struct=struct_name_for_eval)
            )

        elif metric_type == "DoseAtSubVolume":
            if prescription_gy is None:
                raise ValueError("prescription_gy is required for DoseAtSubVolume")
            metric_value = _dose_at_subvolume_percent_rx(
                plan, dose_1d, struct_name_for_eval, subvol_cc=input_value, prescription_gy=prescription_gy
            )

        elif metric_type == "ConformationNumber":
            metric_value = _conformation_number(plan, dose_1d, struct_name_for_eval, ri_dose_gy=input_value)

        else:
            return {
                "Structure": structure_name,
                "MetricType": metric_type,
                "MetricValue": np.nan,
                "Score": np.nan,
                "Status": f"unsupported_metric:{metric_type}",
                "MetricComment": item.get("MetricComment", None),
            }

    except Exception as e:
        return {
            "Structure": structure_name,
            "MetricType": metric_type,
            "MetricValue": np.nan,
            "Score": np.nan,
            "Status": f"metric_error:{e}",
            "MetricComment": item.get("MetricComment", None),
        }

    score = interpolate_score(metric_value, item["ScorePoints"])

    return {
        "Structure": structure_name,
        "MetricType": metric_type,
        "MetricValue": metric_value,
        "Score": score,
        "Status": "ok",
        "MetricComment": item.get("MetricComment", None),
    }


def compute_total_quality_score(plan, dose_1d, scorecard_json, prescription_gy=None, alias_map=None):
    sc = load_scorecard_json(scorecard_json)

    if prescription_gy is None:
        dpf = sc.get("DosePerFraction", None)
        nfx = sc.get("NumberOfFractions", None)
        if dpf is not None and nfx is not None:
            prescription_gy = float(dpf) * float(nfx)

    rows = []
    total_score = 0.0

    for item in sc.get("ScoreTemplates", []):
        row = evaluate_score_item(
            plan=plan,
            dose_1d=dose_1d,
            item=item,
            prescription_gy=prescription_gy,
            alias_map=alias_map
        )
        rows.append(row)
        if not np.isnan(row["Score"]):
            total_score += row["Score"]

    df = pd.DataFrame(rows)
    return total_score, df