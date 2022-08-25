def get_voxels(myPlan, org):
    for i in range(len(myPlan['structures']['Names'])):
        if myPlan['structures']['Names'][i] == org:
            vox = myPlan['structures']['optimizationVoxIndices'][i]

    return vox