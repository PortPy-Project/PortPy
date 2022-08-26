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


def load_metadata(path):
    meta_data = dict()

    fname = os.path.join(path, 'StructureSet_MetaData.json')
    # Opening JSON file
    f = open(fname)
    jsondata = json.load(f)
    meta_data['structures'] = listtodict(jsondata)


    fname = os.path.join(path, 'MachineParams_MetaData.json')
    # Opening JSON file
    f = open(fname)
    jsondata = json.load(f)
    meta_data['machineParams'] = listtodict(jsondata)

    fname = os.path.join(path, 'OptimizationVoxels_MetaData.json')
    # Opening JSON file
    f = open(fname)
    jsondata = json.load(f)
    meta_data['optVoxels'] = listtodict(jsondata)


    fname = os.path.join(path, 'CT_MetaData.json')
    if os.path.isfile(fname):
        # Opening JSON file
        f = open(fname)
        jsondata = json.load(f)
        meta_data['ct'] = listtodict(jsondata)

    fname = os.path.join(path, 'ClinicalCriteria_MetaData.json')
    # Opening JSON file
    f = open(fname)
    jsondata = json.load(f)
    meta_data['clinicalCriteria'] = listtodict(jsondata)

    beamFolder = os.path.join(path, 'Beams')
    beamsJson = [pos_json for pos_json in os.listdir(beamFolder) if pos_json.endswith('.json')]

    beamsJson = natsorted(beamsJson)
    meta_data['beams'] = dict()
    for i in range(len(beamsJson)):
        fname = os.path.join(beamFolder, beamsJson[i])
        f = open(fname)
        jsondata = json.load(f)

        for key in jsondata:
            meta_data['beams'].setdefault(key, []).append(jsondata[key])
            # dataMeta['beamsMetaData'][key].append(jsondata[key])

    meta_data['patientFolderPath'] = path
    return meta_data

