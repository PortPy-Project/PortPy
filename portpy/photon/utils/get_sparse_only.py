import numpy as np
import math
import scipy


def get_sparse_only(A: np.ndarray, threshold_perc: float = 1, compression: str = 'naive', threshold_abs = None):
    """
    Get sparse matrix using threshold and different methods
    :param A: matrix to be sparsified
    :param threshold_perc: threshold for matrix sparsification
    :param compression: Method of Sparsification
    :param threshold_abs: absolute threshold for matrix sparsification

    :return: Sparse influence matrix
    """
    threshold = np.max(A) * threshold_perc * 0.01
    if threshold_abs is not None:
        threshold = threshold_abs
    if compression == 'rmr':
        copy_matrix = A.copy()
        print('Generating sparse matrix using RMR...')
        np.apply_along_axis(row_operation, 1, copy_matrix, threshold)
        S = scipy.sparse.csr_matrix(copy_matrix)
    else:
        S = np.where(A >= threshold, A, 0)
        S = scipy.sparse.csr_matrix(S)
    return S


def row_operation(copy_row, threshold):
    argzero = np.argwhere((np.abs(copy_row) <= threshold) * (copy_row != 0))
    argzero = argzero.reshape(len(argzero), )
    argzero_copy = copy_row[argzero]
    copy_row[argzero] = 0
    sum = np.sum(argzero_copy)
    if sum != 0:
        k = math.ceil(sum / threshold)

        indices = np.random.choice(argzero, k, p=argzero_copy / sum, replace=True)
        np.add.at(copy_row, indices, sum / k)

