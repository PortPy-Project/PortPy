import os
import numpy as np
from typing import List


def get_eclipse_fluence(sol: dict, path: str = None, beam_ids: List[str] = None) -> None:
    """
    save eclipse fluence in the path directory
    """
    if path is None:
        path = os.getcwd()
    tol = 1e-06
    inf_matrix = sol['inf_matrix']
    optimal_fluence_2d = inf_matrix.fluence_1d_to_2d(sol=sol)
    for i in range(len(optimal_fluence_2d)):
        if beam_ids is not None:
            beam_id = beam_ids[i]
        else:
            beam_id = str(inf_matrix.beamlets_dict[i]['beam_id'])
        file_name = 'ID' + beam_id + '.optimal_fluence'
        filepath = os.path.join(path, file_name)
        f = open(filepath, 'w')
        fluence_2d = optimal_fluence_2d[i]
        f.write('optimalfluence\n')
        f.write('SizeX      {}\n'.format(fluence_2d.shape[1]))
        f.write('SizeY      {}\n'.format(fluence_2d.shape[0]))
        f.write('SpacingX   {}\n'.format(2.5))
        f.write('SpacingY   {}\n'.format(2.5))
        beamlets = inf_matrix._beams.beams_dict['beamlets'][i]
        x_positions = beamlets['position_x_mm'][0] - beamlets['width_mm'][
            0] / 2  # x position is center of beamlet. Get left corner
        y_positions = beamlets['position_y_mm'][0] + beamlets['height_mm'][
            0] / 2  # y position is center of beamlet. Get top corner
        f.write('OriginX    {}\n'.format(np.min(x_positions) + 1.25))  # originX should be the center of the first beamlet, but beamlet.X is the left
        f.write('OriginY    {}\n'.format(np.max(y_positions) - 1.25))  # originY should be the center of the first beamlet, but beamlet.Y is the top
        f.write('Values\n')
        f.close()
        fluence_2d[fluence_2d < tol] = 0
        mat = np.matrix(fluence_2d)
        with open(filepath, 'a') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.6f', delimiter='\t')

        # np.savetxt(fileName, optimal_fluence_2d[i], '-append', 'delimiter'='\t', 'precision', '%.10G')

