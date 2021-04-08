#cython: language_level=3
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import cython
import numpy as np
cimport numpy as np
from tqdm import tqdm
from IPython import embed


INT = np.int
BOOL = np.bool
ctypedef np.int_t INT_t
ctypedef np.npy_bool BOOL_t


@cython.boundscheck(False)
@cython.wraparound(False)
def _build_adj_index(np.ndarray[INT_t, ndim=1] values,
                              INT_t max_value):
    # Required: values in ascending order
    cdef INT_t index_size = max_value + 1
    cdef np.ndarray[INT_t, ndim=2] adj_index = np.zeros([index_size, 2], dtype=INT)
    cdef INT_t i = 0, v
    cdef INT_t curr = values[0]

    for i, v in enumerate(values):
        if v != curr:
            curr = v
            adj_index[curr, 0] = i
            adj_index[curr, 1] = i + 1
        else:
            adj_index[curr, 1] += 1

    return adj_index

@cython.wraparound(False)
@cython.boundscheck(False)
def _has_entity_in_component(list stack,
                             np.ndarray[INT_t, ndim=1] to_vertices,
                             np.ndarray[INT_t, ndim=2] adj_index,
                             INT_t num_entities):
    # Perform DFS to look for an entity
    cdef set visited = set()
    cdef bint found = False
    cdef INT_t node
    
    while len(stack) > 0:
        # Pop
        node = stack[len(stack) - 1]
        stack = stack[:len(stack) - 1]

        # Check if node is an entity
        if node < num_entities:
            found = True
            break

        # Skip iteration if node has already been visited
        if node in visited:
            continue
        visited.add(node)

        # Push all nodes reachable from the current node onto the stack
        start_idx, end_idx = adj_index[node, 0], adj_index[node, 1]
        stack.extend(to_vertices[start_idx:end_idx].tolist())

    return found


@cython.boundscheck(False)
@cython.wraparound(False)
def special_partition(np.ndarray[INT_t, ndim=1] row, 
                      np.ndarray[INT_t, ndim=1] col,
                      np.ndarray[INT_t, ndim=1] ordered_indices,
                      np.ndarray[INT_t, ndim=1] siamese_indices,
                      INT_t num_entities,
                      bint directed):
    cdef INT_t num_edges = row.shape[0]
    cdef np.ndarray[BOOL_t, ndim=1] keep_mask = np.ones([num_edges,], dtype=BOOL)
    cdef np.ndarray[INT_t, ndim=1] tmp_graph
    cdef INT_t r, c
    # Flags to track if an entity is reachable from the row or the column, respectively, of the edge that is being dropped
    cdef bint r_entity_reachable, c_entity_reachable = True

    # Build the adjacency matrix for efficient DFS; row-wise for directed, col-wise for undirected
    # Shape [N, 2]; [x,0] to [x,1] (exclusive) is the range of indices for x
    # Example (row adjacency): row X has outgoing edges from X to all values in col[adj_index[X,0]:adj_index[X,1]]
    cdef np.ndarray[INT_t, ndim=2] adj_index
    cdef INT_t max_value = row[len(row) - 1] if directed else col[len(col) - 1] # Last value is max because of sorting
    adj_index = _build_adj_index(row if directed else col, max_value)

    for i in tqdm(ordered_indices, desc='Paritioning joint graph'):
        # Undirected: Skip iteration if the edge has already been dropped
        if keep_mask[i] == False:
            continue

        r, c = row[i], col[i]

        # Remove the forward edge
        keep_mask[i] = False
        # Undirected: Remove the reverse edge
        if not directed:
            keep_mask[siamese_indices[i]] = False

        # Reduce the range by 1 in the adjacency index to reflect the dropped edge
        adj_index[r:, :] -= 1
        adj_index[r, 0] += 1
        if not directed:
            adj_index[c:, :] -= 1
            adj_index[c, 0] += 1

        # Create a temporary graph based on the dropped edge
        tmp_graph = col[keep_mask] if directed else row[keep_mask]

        # Check if an entity can still be reached from r
        r_entity_reachable = _has_entity_in_component(
            [r], tmp_graph, adj_index, num_entities)
        # Undirected: Check if an entity can still be reached from c
        if not directed:
            c_entity_reachable = _has_entity_in_component(
                [c], tmp_graph, adj_index, num_entities)

        # Add (r,c) back if an entity cannot be reached from r or c (when undirected) without it
        if not (r_entity_reachable and c_entity_reachable):
            keep_mask[i] = True
            adj_index[r:, :] += 1
            adj_index[r, 0] -= 1
            if not directed:
                keep_mask[siamese_indices[i]] = True
                adj_index[c:, :] += 1
                adj_index[c, 0] -= 1

    return keep_mask

def cluster_linking_partition(rows, cols, data, n_entities, directed=True):
    assert rows.shape[0] == cols.shape[0] == data.shape[0]
    
    cdef np.ndarray[BOOL_t, ndim=1] keep_edge_mask

    # If undirected, add the reverse edges
    if not directed:
        rows, cols = np.concatenate((rows, cols)), np.concatenate((cols, rows))
        data = np.concatenate((data, data))
    
    # Filter duplicates only on row,col tuples (to accomodate approximation errors in data)
    seen = set()
    _f_row, _f_col, _f_data = [], [], []
    for k in range(len(rows)):
        if (rows[k], cols[k]) in seen:
            continue
        seen.add((rows[k], cols[k]))
        _f_row.append(rows[k])
        _f_col.append(cols[k])
        _f_data.append(data[k])
    rows, cols, data = list(map(np.array, (_f_row, _f_col, _f_data)))

    # Sort data for efficient DFS
    sort_order = lambda x: (x[0], -x[1]) if directed else (x[1], -x[0])
    tuples = zip(rows, cols, data)
    tuples = sorted(tuples, key=sort_order)
    rows, cols, data = zip(*tuples)
    rows = np.asarray(rows, dtype=INT)
    cols = np.asarray(cols, dtype=INT)
    data = np.asarray(data)

    # If undirected, create siamese indices for reverse lookup (i.e. c,r edge to index)
    siamese_idxs = None
    if not directed:
        edge_idxs = {e: i for i, e in enumerate(zip(rows, cols))}
        siamese_idxs = np.array([edge_idxs[(r_c[1], r_c[0])]
                        for r_c in edge_idxs])

    # Order the edges in ascending order of similarity scores
    ordered_edge_idxs = np.argsort(data)

    # Determine which edges to keep in the partitioned graph
    keep_edge_mask = special_partition(
        rows, cols, ordered_edge_idxs, siamese_idxs, n_entities, directed)

    # Return the edges of the partitioned graph
    return rows[keep_edge_mask], cols[keep_edge_mask], data[keep_edge_mask]