import numpy as np

def create_sparse_matrix(ids):
    """
    In
    --
    ids -> (n, 1) numpy array, ideally should be the event_id column

    Out
    ---
    matrix -> (n, n) numpy array, adjacency matrix
    """
    n = len(ids)
    matrix = np.zeros((n, n))

    for row, rid in enumerate(ids):
        for col, cid in enumerate(ids[row+1:]):
            if rid == cid:
                matrix[row, col] = 1

    return matrix
