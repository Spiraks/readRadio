import utils
import numpy as np
import math
from numba import njit
import time


def timer(func):
    def _wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        runtime = time.perf_counter() - start
        print(f"{func.__name__} took {runtime:.4f} secs")
        return result
    return _wrapper


@njit
def rescale(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)


@njit
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


@timer
def h_alpha(path, name, k=1, ver=150, hor=3):
    gg, vv, gv = utils.get_mat(path)
    t11 = compress(np.multiply((gg + vv), np.conj((gg + vv))), ver, hor)
    t12 = compress(np.multiply((gg+vv), np.conj((gg-vv))), ver, hor)
    t13 = compress(np.multiply(2*(gg+vv), np.conj(gv)), ver, hor)
    t21 = compress(np.multiply((gg - vv), np.conj((gg + vv))), ver, hor)
    t22 = compress(np.multiply((gg - vv), np.conj((gg - vv))), ver, hor)
    t23 = compress(np.multiply(2 * (gg - vv), np.conj(gv)), ver, hor)
    t31 = compress(np.multiply(2 * (gg + vv), np.conj(gv)), ver, hor)
    t32 = compress(np.multiply(2 * (gg - vv), np.conj(gv)), ver, hor)
    t33 = compress(np.multiply(4 * gv, np.conj(gv)), ver, hor)
    # utils.draw_rgb(np.absolute(t11), np.absolute(t22), np.absolute(t33), name)

    # Формируем матрицу когерентности
    cog_mat = np.zeros((len(t33), len(t33[0]), 3, 3), dtype=np.complex64)
    t = np.array([t11, t12, t13, t21, t22, t23, t31, t32, t33])
    t = t.reshape((3, 3, len(cog_mat), len(cog_mat[0]))).transpose((2, 3, 0, 1))
    cog_mat = 0.5 * t
    # Вычисляем H и alpha
    eigvals, eigvecs = np.linalg.eig(cog_mat)
    pk = np.absolute(eigvals / np.sum(eigvals, axis=2, keepdims=True))
    h = -(pk * np.log(pk) / np.log(3)).sum(axis=2)
    ca = np.arccos(eigvecs[:, :, 0, :] * np.exp(-1j * np.angle(eigvecs[:, :, 0, :])))
    alpha = (ca * pk).sum(axis=2) * (180 / np.pi)
    # Формируем матрицу цветов
    arr = np.zeros((len(h), len(h[0]), 3))
    arr[:, :, 0] = np.absolute(cog_mat[:, :, 2, 2])
    arr[:, :, 1] = np.absolute(cog_mat[:, :, 1, 1])
    arr[:, :, 2] = np.absolute(cog_mat[:, :, 0, 0])
    arr = np.sqrt(np.absolute(arr))
    mr = np.mean(np.max(arr[:, :, 0], axis=0))
    mg = np.mean(np.max(arr[:, :, 1], axis=0))
    mb = np.mean(np.max(arr[:, :, 2], axis=0))
    arr = k * arr / np.mean([mr, mg, mb])
    arr[arr > 1] = 1
    arr = 255 * arr
    color_cog = rescale(arr)
    color_cog = color_cog.reshape(len(color_cog) * len(color_cog[0]), 3)

    utils.draw_grafics(np.absolute(h), np.absolute(alpha), color_cog, name)


@njit
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
# profiler
# numba jit


if __name__ == '__main__':
    h_alpha('file/pos_data.mat', 'pos_fast', k=4)
    # get_mat('file/led_data.mat', 'led_fast',1)

