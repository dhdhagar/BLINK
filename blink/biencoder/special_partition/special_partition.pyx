#cython: language_level=3
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import cython
import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from tqdm import tqdm
from IPython import embed


INT = np.int
BOOL = np.bool
ctypedef np.int_t INT_t
ctypedef np.float64_t FLOAT_t
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
def _EXP_has_entity_in_component(list stack,
                             FLOAT_t min_score,
                             np.ndarray[INT_t, ndim=1] to_vertices,
                             np.ndarray[FLOAT_t, ndim=1] weights,
                             np.ndarray[INT_t, ndim=2] adj_index,
                             INT_t num_entities,
                             bint dfs,
                             bint get_maximum,
                             bint debug):
    # Perform DFS to look for an entity
    cdef set visited = set()
    cdef bint found = False
    cdef INT_t node
    cdef INT_t immediate_child = -1
    cdef INT_t best_immediate_child = -1
    cdef FLOAT_t path_score = 1
    cdef FLOAT_t node_wt
    cdef FLOAT_t best_found_score = -1
    cdef list weight_stack = [1]
    cdef list prvs_score_stack = []
    cdef list nchild_stack = [1]
    cdef INT_t nchild_popped
    cdef INT_t last_idx
    cdef INT_t go_to_parent = 2 

    while len(stack) > 0:
        if debug:
            print("1")
        # Pop
        if dfs:
            if nchild_stack[len(nchild_stack) - 1] == 0: # go to parent's next sibling
                if debug:
                    print(f'stack={stack}')
                    print(f'prvs_score_stack={prvs_score_stack}')
                    print(f'nchild_stack={nchild_stack}')
                nchild_stack = nchild_stack[:len(nchild_stack) - 1]
                if debug:
                    print("1a")
                path_score = prvs_score_stack[len(prvs_score_stack) - go_to_parent]
                if debug:
                    print("1b")
                prvs_score_stack = prvs_score_stack[:len(prvs_score_stack) - go_to_parent]
                if debug:
                    print("1c")
            if debug:
                print("2")
            # DFS
            last_idx = len(stack) - 1
            node = stack[last_idx]
            # Track the immediate edge from the root of the current DFS path being evaluated
            if len(nchild_stack) == 2:
                immediate_child = node
            stack = stack[:last_idx]
            node_wt = weight_stack[last_idx]
            weight_stack = weight_stack[:last_idx]
            # Skip iteration if node has already been visited
            if node in visited:
                nchild_stack[len(nchild_stack) - 1] = nchild_stack[len(nchild_stack) - 1] - 1
                go_to_parent = 1
                continue

            if path_score * node_wt < max(min_score, best_found_score):
                nchild_stack[len(nchild_stack) - 1] = nchild_stack[len(nchild_stack) - 1] - 1
                go_to_parent = 1
                continue
            else:
                prvs_score_stack.append(path_score)
                path_score = path_score * node_wt
                go_to_parent = 2
            if debug:
                print("3")
        else:
            # BFS
            node = stack[0]
            stack = stack[1:]

        # Check if node is an entity
        if node < num_entities:
            best_found_score = path_score
            best_immediate_child = immediate_child
            if debug:
                print("4")
            if not get_maximum:
                return best_found_score, best_immediate_child

        visited.add(node)
        if debug:
            print("5")
        # Push all nodes reachable from the current node onto the stack
        start_idx, end_idx = adj_index[node, 0], adj_index[node, 1]
        next_nodes = to_vertices[start_idx:end_idx].tolist()
        if len(next_nodes) != 0:
            stack.extend(next_nodes)
            weight_stack.extend(weights[start_idx:end_idx].tolist())
            nchild_stack.append(len(next_nodes))
            if debug:
                print("6")
        else:
            nchild_stack[len(nchild_stack) - 1] = nchild_stack[len(nchild_stack) - 1] - 1
            if nchild_stack[len(nchild_stack) - 1] != 0:
                path_score = prvs_score_stack[len(prvs_score_stack) - 1]
                prvs_score_stack = prvs_score_stack[:len(prvs_score_stack) - 1]
            if debug:
                print("7")
    return best_found_score, best_immediate_child

@cython.wraparound(False)
@cython.boundscheck(False)
def _has_entity_in_component(list stack,
                             np.ndarray[INT_t, ndim=1] to_vertices,
                             np.ndarray[INT_t, ndim=2] adj_index,
                             INT_t num_entities,
                             bint dfs):
    # Perform DFS to look for an entity
    cdef set visited = set()
    cdef bint found = False
    cdef INT_t node
    
    while len(stack) > 0:
        # Pop
        if dfs:
            # DFS
            node = stack[len(stack) - 1]
            stack = stack[:len(stack) - 1]
        else:
            # BFS
            node = stack[0]
            stack = stack[1:]

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
def _EXP_special_partition(np.ndarray[INT_t, ndim=1] row, 
                      np.ndarray[INT_t, ndim=1] col,
                      np.ndarray[FLOAT_t, ndim=1] data,
                      np.ndarray[INT_t, ndim=1] ordered_indices,
                      np.ndarray[INT_t, ndim=1] siamese_indices,
                      dict edge_indices,
                      INT_t num_entities,
                      bint directed,
                      bint dfs,
                      bint silent):
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
    cdef bint reachable_from_edge
    cdef FLOAT_t score_from_edge
    cdef FLOAT_t r_entity_score
    
    # Scale data to [0, 1]
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    adj_index = _build_adj_index(row if directed else col, max_value)

    if silent:
        _iter = ordered_indices
    else:
        _iter = tqdm(ordered_indices, desc='Paritioning joint graph')
    
    for i in _iter:
        # Undirected: Skip iteration if the edge has already been dropped
        if keep_mask[i] == False:
            continue

        r, c, edge_wt = row[i], col[i], data[i]

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
        # Check score of path from current edge to an entity
        reachable_from_edge = _has_entity_in_component(
            [c], tmp_graph, adj_index, num_entities, dfs)
        score_from_edge, _ = _EXP_has_entity_in_component(
                [c], -1, tmp_graph, data[keep_mask], adj_index, num_entities, dfs, True, False) # r==2344431 and c==2340167
        assert (not reachable_from_edge and score_from_edge == -1) or (reachable_from_edge and score_from_edge != -1)
        # Check if an entity can still be reached from r and that has a path score greater than the current path score
        r_entity_reachable = _has_entity_in_component([r], tmp_graph, adj_index, num_entities, dfs)
        if not reachable_from_edge and not r_entity_reachable:
            print("Unreachable with and without")
        r_entity_score, _ = _EXP_has_entity_in_component(
            [r], -1, tmp_graph, data[keep_mask], adj_index, num_entities, dfs, False, False)
        assert (not r_entity_reachable and r_entity_score == -1) or (r_entity_reachable and r_entity_score != -1)
        r_entity_score, best_path_node = _EXP_has_entity_in_component(
            [r], edge_wt * score_from_edge, tmp_graph, data[keep_mask], adj_index, num_entities, dfs, False, False)
        r_entity_reachable = r_entity_score != -1
            
        # Undirected: Check if an entity can still be reached from c
        # if not directed:
        #     c_entity_reachable = _EXP_has_entity_in_component(
        #         [c], tmp_graph, adj_index, num_entities, dfs)

        # Add (r,c) back if an entity cannot be reached from r or c (when undirected) without it
        if not (r_entity_reachable and c_entity_reachable):
            keep_mask[i] = True
            adj_index[r:, :] += 1
            adj_index[r, 0] -= 1
            # Remove the best edge in the temporary graph that was rejected
            if best_path_node != -1:
                adj_index[r:, :] -= 1
                adj_index[r, 0] += 1
                keep_mask[edge_indices[(r,best_path_node)]] = False
            if not directed:
                keep_mask[siamese_indices[i]] = True
                adj_index[c:, :] += 1
                adj_index[c, 0] -= 1

    return keep_mask

@cython.boundscheck(False)
@cython.wraparound(False)
def special_partition(np.ndarray[INT_t, ndim=1] row, 
                      np.ndarray[INT_t, ndim=1] col,
                      np.ndarray[INT_t, ndim=1] ordered_indices,
                      np.ndarray[INT_t, ndim=1] siamese_indices,
                      INT_t num_entities,
                      bint directed,
                      bint dfs,
                      bint silent):
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

    if silent:
        _iter = ordered_indices
    else:
        _iter = tqdm(ordered_indices, desc='Paritioning joint graph')
    
    for i in _iter:
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
            [r], tmp_graph, adj_index, num_entities, dfs)
        # Undirected: Check if an entity can still be reached from c
        if not directed:
            c_entity_reachable = _has_entity_in_component(
                [c], tmp_graph, adj_index, num_entities, dfs)

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

def cluster_linking_partition(rows, cols, data, n_entities, directed=True, dfs=True, silent=False, exclude=set(), threshold=None, experimental=False):
    assert rows.shape[0] == cols.shape[0] == data.shape[0]
    
    cdef np.ndarray[BOOL_t, ndim=1] keep_edge_mask

    # Filter duplicates only on row,col tuples (to accomodate approximation errors in data)
    # Additionally, filter out any nodes passed in 'exclude'
    seen = set()
    duplicated, excluded, thresholded = 0, 0, 0
    _f_row, _f_col, _f_data = [], [], []
    for k in range(len(rows)):
        if (rows[k], cols[k]) in seen:
            duplicated += 1
            continue
        seen.add((rows[k], cols[k]))
        if rows[k] in exclude or cols[k] in exclude:
            excluded += 1
            continue
        if threshold is not None and data[k] < threshold:
            thresholded += 1
            continue
        _f_row.append(rows[k])
        _f_col.append(cols[k])
        _f_data.append(data[k])
    rows, cols, data = list(map(np.array, (_f_row, _f_col, _f_data)))

    if duplicated + excluded + thresholded > 0:
        print(f"""
Dropped edges during pre-processing:
    Duplicates: {duplicated}
    Excluded: {excluded}
    Thresholded: {thresholded}""")

    if not directed:
        # Filter down using Scipy's MST routine for faster processing
        shape = int(max(np.max(rows), np.max(cols))) + 1
        shape = (shape, shape)
        csr = csr_matrix((-data, (rows, cols)), shape=shape)
        mst = minimum_spanning_tree(csr).tocoo()
        rows, cols, data = mst.row, mst.col, -mst.data
    
        # Add the reverse edges
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
    sort_order = lambda x: (x[0], x[1]*(-1 if dfs else 1)) if directed else (x[1], x[0]*(-1 if dfs else 1)) # For faster iterations: descending order for DFS, ascending order for BFS
    tuples = zip(rows, cols, data)
    tuples = sorted(tuples, key=sort_order)
    rows, cols, data = zip(*tuples)
    rows = np.asarray(rows, dtype=INT)
    cols = np.asarray(cols, dtype=INT)
    data = np.asarray(data)

    # If undirected, create siamese indices for reverse lookup (i.e. c,r edge to index)
    cdef dict edge_idxs = {e: i for i, e in enumerate(zip(rows, cols))}
    siamese_idxs = np.array([edge_idxs[(r_c[1], r_c[0])] 
                            if (r_c[1], r_c[0]) in edge_idxs else -1
                            for r_c in edge_idxs])

    # Order the edges in ascending order of similarity scores
    ordered_edge_idxs = np.argsort(data)

    # Determine which edges to keep in the partitioned graph
    if experimental:
        keep_edge_mask = _EXP_special_partition(
            rows, cols, data, ordered_edge_idxs, siamese_idxs, edge_idxs, n_entities, directed, dfs, silent)
    else:
        keep_edge_mask = special_partition(
            rows, cols, ordered_edge_idxs, siamese_idxs, n_entities, directed, dfs, silent)

    # Return the edges of the partitioned graph
    return rows[keep_edge_mask], cols[keep_edge_mask], data[keep_edge_mask]