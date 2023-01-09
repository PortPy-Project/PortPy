import json
import os
from natsort import natsorted

import itertools


def listtodict(jsondata):
    '''
    A recursive function which constructs from dictionary from list of dictionary
    '''
    jsondict = {}
    if type(jsondata) is list:
        for i in range(len(jsondata)):
            elem = jsondata[i]
            if type(elem) is list:
                jsondict[i] = listtodict(elem)
            else:
                for key in elem:
                    jsondict.setdefault(key, []).append(elem[key])
    else:
        jsondict = jsondata.copy()
    return jsondict


def load_metadata(path: str, options=None) -> dict:
    """Loads metadata of a patient located in path and returns the metadata as a dictionary

    :param path: full path of patient folder
    :param options:
    :return: a dictionary including all metadata
    """

    meta_data = dict()  # initialization
    """
    The data are loaded from the following .Json files:
    1- StructureSet_MetaData.json
        including data about the structures (e.g., PTV, Kidney, Lung)
    2- OptimizationVoxels_MetaData.json
        including patient voxel data (3D cubic voxels of patient body)
    3- CT_MetaData.json
        including patient CT scan data (e.g., size, resolution, ct hounsfield units)
    4- PlannerBeams.json
        including the indices of the beams selected by an expert planner based on the geometry/shape/location of tumor/healthy-tissues 
    5- ClinicalCriteria_MetaData.json
        including clinically relevant metrics used to evaluate a plan (e.g., Kidney mean dose <= 20Gy, Cord max dose <= 10 Gy)
    6- Beams.json 
        including beam information (e.g., gantry angle, collimator angle)
    """

    # read information regarding the structures
    fname = os.path.join(path, 'StructureSet_MetaData.json')
    # Opening JSON file
    f = open(fname)
    jsondata = json.load(f)
    meta_data['structures'] = listtodict(jsondata)

    # read information regarding the voxels
    fname = os.path.join(path, 'OptimizationVoxels_MetaData.json')
    # Opening JSON file
    f = open(fname)
    jsondata = json.load(f)
    meta_data['opt_voxels'] = listtodict(jsondata)

    # read information regarding the CT voxels
    fname = os.path.join(path, 'CT_MetaData.json')
    if os.path.isfile(fname):
        # Opening JSON file
        f = open(fname)
        jsondata = json.load(f)
        meta_data['ct'] = listtodict(jsondata)

    # read information regarding beam angles selected by an expert planner
    fname = os.path.join(path, 'PlannerBeams.json')
    if os.path.isfile(fname):
        # Opening JSON file
        f = open(fname)
        jsondata = json.load(f)
        meta_data['planner_beam_ids'] = listtodict(jsondata)

    # read information regarding the clinical evaluation metrics
    fname = os.path.join(path, 'ClinicalCriteria_MetaData.json')
    # Opening JSON file
    f = open(fname)
    jsondata = json.load(f)
    meta_data['clinical_criteria'] = listtodict(jsondata)

    # read information regarding the beams
    beamFolder = os.path.join(path, 'Beams')
    beamsJson = [pos_json for pos_json in os.listdir(beamFolder) if pos_json.endswith('.json')]

    beamsJson = natsorted(beamsJson)
    meta_data['beams'] = dict()
    # the information for each beam is stored in an individual .json file, so we loop through them
    for i in range(len(beamsJson)):
        fname = os.path.join(beamFolder, beamsJson[i])
        f = open(fname)
        jsondata = json.load(f)

        for key in jsondata:
            meta_data['beams'].setdefault(key, []).append(jsondata[key])
            # dataMeta['beamsMetaData'][key].append(jsondata[key])

    meta_data['patient_folder_path'] = path
    meta_data = load_options(options=options, meta_data=meta_data)
    return meta_data


def load_options(options=None, meta_data=None):
    if options is None:
        options = dict()
        options['loadInfluenceMatrixFull'] = 0
        options['loadInfluenceMatrixSparse'] = 1
    if len(options) != 0:
        if 'loadInfluenceMatrixFull' in options and not options['loadInfluenceMatrixFull']:
            meta_data['beams']['influenceMatrixFull_File'] = [None] * len(
                meta_data['beams']['influenceMatrixSparse_File'])
        if 'loadInfluenceMatrixSparse' in options and not options['loadInfluenceMatrixSparse']:
            meta_data['beams']['influenceMatrixFull_File'] = [None] * len(
                meta_data['beams']['influenceMatrixSparse_File'])
    return meta_data