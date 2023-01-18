from portpy.influence_matrix import InfluenceMatrix


def sol_change_inf_matrix(sol: dict, inf_matrix: InfluenceMatrix) -> dict:
    """
    Create a new solution by changing the basis of current solution.
    It will create a solution with same number of beamlets and voxels as inf_matrix


    :param sol: solution for which influence matrix is changed
    :param inf_matrix: object of class Influence matrix
    :return: new solution dictionary having same number of beamlets and voxels as inf_matrix
    """
    new_sol = dict()
    if sol['optimal_intensity'].shape[0] < inf_matrix.A.shape[1]:
        optimal_intensity = sol['inf_matrix'].fluence_1d_to_2d(fluence_1d=sol['optimal_intensity'])
        new_sol['optimal_intensity'] = inf_matrix.fluence_2d_to_1d(optimal_intensity)
    elif sol['optimal_intensity'].shape[0] == inf_matrix.A.shape[1]:
        new_sol['optimal_intensity'] = sol['optimal_intensity']
    else:
        raise ValueError("Beamlet resolution should be greater than or equal to beamlets for inf_matrix")

    # new_sol['dose_1d'] = inf_matrix.A * new_sol['optimal_intensity']

    # dose_3d = sol['inf_matrix'].dose_1d_to_3d(dose_1d=sol['dose_1d'])
    # new_sol['dose_1d'] = inf_matrix.dose_3d_to_1d(dose_3d=dose_3d)

    new_sol['inf_matrix'] = inf_matrix
    return new_sol


