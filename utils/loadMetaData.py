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

def loadMetaData(path):
    metaData = dict()

    fname = os.path.join(path, 'StructureSet_MetaData.json')
    # Opening JSON file
    f = open(fname)
    jsondata = json.load(f)
    metaData['structures'] = listtodict(jsondata)


    fname = os.path.join(path, 'MachineParams_MetaData.json')
    # Opening JSON file
    f = open(fname)
    jsondata = json.load(f)
    metaData['machineParams'] = listtodict(jsondata)

    fname = os.path.join(path, 'OptimizationVoxels_MetaData.json')
    # Opening JSON file
    f = open(fname)
    jsondata = json.load(f)
    metaData['optVoxels'] = listtodict(jsondata)


    fname = os.path.join(path, 'CT_MetaData.json')
    if os.path.isfile(fname):
        # Opening JSON file
        f = open(fname)
        jsondata = json.load(f)
        metaData['ct'] = listtodict(jsondata)

    fname = os.path.join(path, 'ClinicalCriteria_MetaData.json')
    # Opening JSON file
    f = open(fname)
    jsondata = json.load(f)
    metaData['clinicalCriteria'] = listtodict(jsondata)

    beamFolder = os.path.join(path, 'Beams')
    beamsJson = [pos_json for pos_json in os.listdir(beamFolder) if pos_json.endswith('.json')]

    beamsJson = natsorted(beamsJson)
    metaData['beams'] = dict()
    for i in range(len(beamsJson)):
        fname = os.path.join(beamFolder, beamsJson[i])
        f = open(fname)
        jsondata = json.load(f)

        for key in jsondata:
            metaData['beams'].setdefault(key, []).append(jsondata[key])
            # dataMeta['beamsMetaData'][key].append(jsondata[key])

    metaData['patientFolderPath'] = path
    return metaData


if __name__ == "__main__":
    path = r'\\pisidsmph\Treatplanapp\ECHO\Research\Data_newformat\PatientData\Lung_Patient_1'
    metaData = loadMetaData(path)
