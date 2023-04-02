import utils
import numpy as np
import math
# import numba as nb
import time
import my_stat


def timer(func):
    def _wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        runtime = time.perf_counter() - start
        print(f"{func.__name__} took {runtime:.4f} secs")
        return result
    return _wrapper


@timer
def commress_elem(gg, vv, gv, ver, hor):
    t11 = my_stat.compress(np.multiply((gg + vv), np.conj((gg + vv))), ver, hor)
    t12 = my_stat.compress(np.multiply((gg + vv), np.conj((gg - vv))), ver, hor)
    t13 = my_stat.compress(np.multiply(2 * (gg + vv), np.conj(gv)), ver, hor)
    t21 = my_stat.compress(np.multiply((gg - vv), np.conj((gg + vv))), ver, hor)
    t22 = my_stat.compress(np.multiply((gg - vv), np.conj((gg - vv))), ver, hor)
    t23 = my_stat.compress(np.multiply(2 * (gg - vv), np.conj(gv)), ver, hor)
    t31 = my_stat.compress(np.multiply(2 * (gg + vv), np.conj(gv)), ver, hor)
    t32 = my_stat.compress(np.multiply(2 * (gg - vv), np.conj(gv)), ver, hor)
    t33 = my_stat.compress(np.multiply(4 * gv, np.conj(gv)), ver, hor)
    return [t11, t12, t13, t21, t22, t23, t31, t32, t33]

@timer
def create_cog_arr(t):
    cog_mat = np.zeros((len(t[8]), len(t[8][0]), 3, 3), dtype=np.complex64)
    return 0.5 * t.reshape((3, 3, len(cog_mat), len(cog_mat[0]))).transpose((2, 3, 0, 1))

@timer
def h_alpha(cog_mat):
    eigvals, eigvecs = np.linalg.eig(cog_mat)
    pk = np.absolute(eigvals / np.sum(eigvals, axis=2, keepdims=True))
    h = -(pk * np.log(pk) / np.log(3)).sum(axis=2)
    ca = np.arccos(eigvecs[:, :, 0, :] * np.exp(-1j * np.angle(eigvecs[:, :, 0, :])))
    alpha = (ca * pk).sum(axis=2) * (180 / np.pi)
    return h, alpha

@timer
def color_matrix(cog_mat, k):
    arr = np.zeros((len(cog_mat), len(cog_mat[0]), 3))
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
    color_cog = my_stat.rescale(arr)
    return color_cog.reshape(len(color_cog) * len(color_cog[0]), 3)

@timer
def decompoz(path, name, k=1, ver=150, hor=3):
    gg, vv, gv = utils.get_mat(path)
    t = np.array(commress_elem(gg, vv, gv, ver, hor))
    utils.draw_rgb(np.absolute(t[0]), np.absolute(t[4]), np.absolute(t[8]), name)
    # Формируем матрицу когерентности
    cog_mat = create_cog_arr(t)
    # Вычисляем H и alpha
    h, alpha = h_alpha(cog_mat)
    # Формируем матрицу цветов
    color_mat = color_matrix(cog_mat,k)
    utils.draw_grafics(np.absolute(h), np.absolute(alpha), color_mat, name)

# profiler
# numba jit


if __name__ == '__main__':
    decompoz('file/pos_data.mat', 'pos_fast', k=4)
    # decompoz('file/led_data.mat', 'led_fast',1)

