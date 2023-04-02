import numpy as np
import math
import numba as nb


@nb.njit
def compress(original, ver, hor):
    rows, cols = original.shape
    new_rows = math.ceil(rows / ver)
    new_cols = math.ceil(cols / hor)

    current_matrix = np.zeros((new_rows, new_cols), dtype=np.complex64)
    for i in range(new_rows):
        for j in range(new_cols):
            row_start = i * ver
            row_end = min(row_start + ver, rows)
            col_start = j * hor
            col_end = min(col_start + hor, cols)
            current_matrix[i, j] = np.sum(original[row_start:row_end, col_start:col_end])

    return current_matrix


def rescale(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)


def max_axis(arr, k=1):
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    mr = np.mean(np.max(red, axis=0))
    mg = np.mean(np.max(green, axis=0))
    mb = np.mean(np.max(blue, axis=0))
    arr = k * arr / np.mean([mr, mg, mb])
    arr[arr > 1] = 1
    arr = 255 * arr
    return arr