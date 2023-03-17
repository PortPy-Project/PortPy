import json
import os
from natsort import natsorted
from pathlib import Path


def list_to_dict(json_data):
    """
    A recursive function which constructs dictionary from list
    :param json_data: data in json or list format
    :return: data in dictionary format
    """

    json_dict = {}
    if type(json_data) is list:
        for i in range(len(json_data)):
            elem = json_data[i]
            if type(elem) is list:
                json_dict[i] = list_to_dict(elem)
            else:
                for key in elem:
                    json_dict.setdefault(key, []).append(elem[key])
    else:
        json_dict = json_data.copy()
    return json_dict


def load_metadata(pat_dir: str) -> dict:
    """Loads metadata of a patient located in path and returns the metadata as a dictionary

    The data are loaded from the following .Json files:
    1- StructureSet_MetaData.json
        including data about the structures (e.g., PTV, Kidney, Lung)
    2- OptimizationVoxels_MetaData.json
        including patient voxel data (3D cubic voxels of patient body)
    3- CT_MetaData.json
        including patient CT scan data (e.g., size, resolution, ct hounsfield units)
    4- PlannerBeams.json
        including the indices of the beams_dict selected by an expert planner based on the geometry/shape/location of tumor/healthy-tissues
    5- ClinicalCriteria_MetaData.json
        including clinically relevant metrics used to evaluate a plan (e.g., Kidney mean dose_1d <= 20Gy, Cord max dose_1d <= 10 Gy)
    6- Beams.json
        including beam information (e.g., gantry angle, collimator angle)

    :param pat_dir: full path of patient folder
    :return: a dictionary including all metadata
    """

    meta_data = dict()  # initialization

    # read information regarding the structures
    fname = os.path.join(pat_dir, 'StructureSet_MetaData.json')
    # Opening JSON file
    f = open(fname)
    jsondata = json.load(f)
    meta_data['structures'] = list_to_dict(jsondata)

    # read information regarding the voxels
    fname = os.path.join(pat_dir, 'OptimizationVoxels_MetaData.json')
    # Opening JSON file
    f = open(fname)
    jsondata = json.load(f)
    meta_data['opt_voxels'] = list_to_dict(jsondata)

    # read information regarding the CT voxels
    fname = os.path.join(pat_dir, 'CT_MetaData.json')
    if os.path.isfile(fname):
        # Opening JSON file
        f = open(fname)
        jsondata = json.load(f)
        meta_data['ct'] = list_to_dict(jsondata)

    # read information regarding beam angles selected by an expert planner
    fname = os.path.join(pat_dir, 'PlannerBeams.json')
    if os.path.isfile(fname):
        # Opening JSON file
        f = open(fname)
        jsondata = json.load(f)
        meta_data['planner_beam_ids'] = list_to_dict(jsondata)

    # read information regarding the clinical evaluation metrics
    fname = os.path.join(pat_dir, 'ClinicalCriteria_MetaData.json')
    # Opening JSON file
    f = open(fname)
    jsondata = json.load(f)
    meta_data['clinical_criteria'] = list_to_dict(jsondata)

    # read information regarding the beams_dict
    beamFolder = os.path.join(pat_dir, 'Beams')
    beamsJson = [pos_json for pos_json in os.listdir(beamFolder) if pos_json.endswith('.json')]

    beamsJson = natsorted(beamsJson)
    meta_data['beams_dict'] = dict()
    # the information for each beam is stored in an individual .json file, so we loop through them
    for i in range(len(beamsJson)):
        fname = os.path.join(beamFolder, beamsJson[i])
        f = open(fname)
        jsondata = json.load(f)

        for key in jsondata:
            meta_data['beams_dict'].setdefault(key, []).append(jsondata[key])
            # dataMeta['beamsMetaData'][key].append(json_data[key])

    meta_data['patient_folder_path'] = pat_dir
    return meta_data


def load_config_planner_metadata(patient_id):
    # load planner_plan config metadata
    fname = os.path.join(Path(__file__).parents[2], 'config_files', 'planner_plan', patient_id, 'planner_plan.json')
    # fname = os.path.join('..', 'config_files', 'planner_plan', patient_id, 'planner_plan.json')
    # Opening JSON file
    f = open(fname)
    planner_metadata = json.load(f)
    return planner_metadata


def load_config_clinical_criteria(protocol_type, protocol_name):
    # load planner_plan config metadata
    fname = os.path.join(Path(__file__).parents[2], 'config_files', 'clinical_criteria',
                         protocol_type, protocol_name + '.json')
    # fname = os.path.join('..', 'config_files', 'planner_plan', patient_id, 'planner_plan.json')
    # Opening JSON file
    f = open(fname)
    metadata = json.load(f)
    return metadata
