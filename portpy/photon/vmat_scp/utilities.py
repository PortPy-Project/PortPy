import sys
from warnings import warn

import numpy as np
import scipy as sp
import scipy.sparse as sparse
from collections import defaultdict
# Lazy sparse_dot_mkl: importing it eagerly loads the Intel MKL runtime, which on Windows
# breaks a subsequent `import torch` (OSError WinError 127, torch\\lib\\shm.dll). Deferring the
# import to first use keeps `import portpy.photon` torch-safe. Resolved once, then cached.
_dot_mkl_impl = None
def dot_mkl(A, B):
    global _dot_mkl_impl
    if _dot_mkl_impl is None:
        try:
            from sparse_dot_mkl import dot_product_mkl as _impl
        except ImportError:
            warn("sparse_dot_mkl not found. Falling back to scipy.sparse dot product, which may be slower.")
            def _impl(A, B):
                return A.dot(B)
        _dot_mkl_impl = _impl
    return _dot_mkl_impl(A, B)

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

# Column-wise sum of each list of A column indices specified in col_list.
# For r, ind_list in enumerate(col_list), calculate B_col_list[r] = np.sum(A[:,ind_list], axis=1). Result is np.column_stack(B_col_list)

def sum_col_list(A, col_list, todense = True):
    num_col_sum = len(col_list)
    matsum_row_ind = np.array([rind for rind_list in col_list for rind in rind_list])
    if len(matsum_row_ind) == 0:
        return np.zeros((A.shape[0], num_col_sum)) if todense else sparse.csr_matrix((A.shape[0], num_col_sum))

    num_col_ind = len(matsum_row_ind)
    # matsum_col_ind_list = [[r]*len(ind_list) for r, ind_list in enumerate(col_list)]
    # matsum_col_ind = np.array([cind for cind_list in matsum_col_ind_list for cind in cind_list])
    matsum_col_ind_list = [np.repeat(r, len(ind_list)) for r, ind_list in enumerate(col_list)]
    matsum_col_ind = np.concatenate(matsum_col_ind_list)
    matsum_data = np.ones(num_col_ind, dtype=A.dtype)
    matsum = sparse.csr_matrix((matsum_data, (matsum_row_ind, matsum_col_ind)), shape=(A.shape[1], num_col_sum))
    A_col_sums = dot_mkl(A, matsum)
    return A_col_sums.todense() if todense else A_col_sums

# Create dict mapping matrix element (key) to column index (value).
def create_elem_to_col_dict(B):
    d = defaultdict(list)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            d[B[i,j]].append(j)
    return d

def get_first_col_match(B, elem_list, default_val = np.nan):
    B_dict = create_elem_to_col_dict(B)
    return get_first_col_match_from_lookup(B_dict, elem_list, default_val = default_val)

def get_first_col_match_from_lookup(B_dict, elem_list, default_val = np.nan):
    v = np.zeros(len(elem_list))
    for i, elem in enumerate(elem_list):
        v[i] = default_val if len(B_dict[elem]) == 0 else B_dict[elem][0]
    return v
