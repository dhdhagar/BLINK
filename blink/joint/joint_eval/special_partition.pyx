#cython: language_level=3
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import cython
import numpy as np
cimport numpy as np
from tqdm import tqdm

INT = np.int
BOOL = np.bool

ctypedef np.int_t INT_t
ctypedef np.npy_bool BOOL_t


@cython.boundscheck(False)
@cython.wraparound(False)
def _build_col_wise_adj_index(np.ndarray[INT_t, ndim=1] col,
                              INT_t col_max_value):
    # requires: sorted col ascending order
    cdef INT_t index_size = col_max_value + 1
    cdef np.ndarray[INT_t, ndim=2] col_wise_adj_index = np.zeros([index_size, 2], dtype=INT)
    cdef INT_t adjusted_index
    cdef INT_t i = 0, c

    cdef INT_t curr_col = col[0]
    for i, c in enumerate(col):
        if c != curr_col:
            curr_col = c
            col_wise_adj_index[curr_col, 0] = i
            col_wise_adj_index[curr_col, 1] = i+1
        else:
            col_wise_adj_index[curr_col, 1] += 1

    return col_wise_adj_index


@cython.boundscheck(False)
def _has_entity_in_component(list stack,
                             np.ndarray[INT_t, ndim=1] row,
                             np.ndarray[INT_t, ndim=2] col_wise_adj_index,
                             INT_t num_entities):
    # performs DFS and returns `True` whenever it hits an entity
    cdef bint has_entity = False
    cdef INT_t index_size = col_wise_adj_index.shape[0]
    cdef set visited = set()
    cdef INT_t curr_node

    while len(stack) > 0:
        # pop
        curr_node = stack[-1]
        stack = stack[:-1]

        # check if `curr_node` is an entity
        if curr_node < num_entities:
            has_entity = True
            break

        # check if we've visited `curr_node`
        if curr_node in visited:
            continue
        visited.add(curr_node)

        # get neighbors of `curr_node` and push them onto the stack
        start_index = col_wise_adj_index[curr_node, 0]
        end_index = col_wise_adj_index[curr_node, 1]
        stack.extend(row[start_index:end_index].tolist())
    
    return has_entity


@cython.boundscheck(False)
@cython.wraparound(False)
def special_partition(np.ndarray[INT_t, ndim=1] row, 
                      np.ndarray[INT_t, ndim=1] col,
                      np.ndarray[INT_t, ndim=1] ordered_indices,
                      np.ndarray[INT_t, ndim=1] siamese_indices,
                      INT_t num_entities):
    assert row.shape[0] == col.shape[0]
    assert row.shape[0] == ordered_indices.shape[0]
    assert row.shape[0] == siamese_indices.shape[0]

    cdef INT_t num_edges = row.shape[0]
    cdef np.ndarray[BOOL_t, ndim=1] keep_mask = np.ones([num_edges,], dtype=BOOL)
    cdef np.ndarray[INT_t, ndim=1] tmp_row, tmp_col
    cdef INT_t r, c
    cdef bint has_entity_r, has_entity_c
    cdef INT_t col_max_value = np.max(col)

    # has shape [N, 2]; [:,0] are starting indices and [:,1] are (exclusive) ending indices
    cdef np.ndarray[INT_t, ndim=2] col_wise_adj_index
    cdef INT_t adjusted_index
    col_wise_adj_index = _build_col_wise_adj_index(
            col, col_max_value
    )

    for i in tqdm(ordered_indices, desc='Paritioning Joint Graph'):
        r = row[i]
        c = col[i]

        # we've already deleted this edge so we can move on
        if keep_mask[i] == False:
            continue

        # try removing both the forward and backward edges
        keep_mask[i] = False
        keep_mask[siamese_indices[i]] = False

        # update the adj list index for the forward and backward edges
        col_wise_adj_index[c:, :] -= 1
        col_wise_adj_index[c, 0] += 1
        col_wise_adj_index[r:, :] -= 1
        col_wise_adj_index[r, 0] += 1

        # create the temporary graph we want to check
        tmp_row = row[keep_mask]

        # check if we can remove the edge (r, c) 
        has_entity_r = _has_entity_in_component(
                [r], tmp_row, col_wise_adj_index, num_entities
        )
        has_entity_c = _has_entity_in_component(
                [c], tmp_row, col_wise_adj_index, num_entities
        )

        # add the edge back if we need it
        if not(has_entity_r and has_entity_c):
            keep_mask[i] = True
            keep_mask[siamese_indices[i]] = True
            col_wise_adj_index[c:, :] += 1
            col_wise_adj_index[c, 0] -= 1
            col_wise_adj_index[r:, :] += 1
            col_wise_adj_index[r, 0] -= 1

    return keep_mask
