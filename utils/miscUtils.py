import numpy as np

# Scale dose vector so V(vol_perc%) = p, i.e., vol_perc% of voxels receive 100% of prescribed dose p.
def scaleDose(d, p, vol_perc):
    d_perc = np.percentile(d, 1 - vol_perc)
    scale = p / d_perc
    return scale*d
