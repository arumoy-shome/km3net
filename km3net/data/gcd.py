import numpy as np

def get_label(relations):
    """
    In
    --
    relations -> (n, 1) numpy array, ideally should be the output of the mlp

    Out
    ---
    label -> Int, label of the sample
    """
    return np.amax(relations)

def process(relations, olen):
    """
    In
    --
    relations -> (n, 1) numpy array, ideally should be the output of the mlp
    olen -> Int, original len of dataset (input of km3net.data.pm.process)

    Out
    ---
    matrix -> (n, n) numpy array, adjacency matrix
    """
    matrix = np.zeros((olen, olen))
    start = 0
    window = olen - 1
    end = start + window

    for row in range(olen):
        col = row + 1
        matrix[row, col:] = relations[start:end]
        start = end
        window -= 1
        end = start + window

    return matrix, get_label(relations)
