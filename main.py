import scipy.io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from PIL import Image
import math
import matplotlib.colors as mcolors


def rescale(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)

def norm_3_4(arr):
    arr_flat = arr.flatten()
    arr_sort = np.array(sorted(arr_flat))
    print('len = ', np.shape(arr), '; len_sort = ', np.shape(arr_sort))
    lengh = len(arr_sort)
    arr_sort = arr_sort[int(lengh * 3 / 4):lengh]
    print('len = ', np.shape(arr), '; len_sort = ', np.shape(arr_sort))
    median2 = np.mean(arr_sort) * 3
    print('max = ', arr.max(), '; max_sort = ', arr_sort.max())
    print('min = ', arr.min(), '; min_sort = ', arr_sort.min())
    arr[arr > median2] = median2
    arr = 255.0 * rescale(arr)
    return arr
def norm_1_4(arr):
    arr_flat = arr.flatten()
    arr_sort = np.array(sorted(arr_flat))
    lengh = len(arr_sort)
    arr_sort = arr_sort[:int(lengh * 1 / 2)]
    median2 = np.max(arr_sort)*4
    arr[arr < median2] = median2
    arr = 255.0 * rescale(arr)
    return arr

def max_axis(arr, k=1):
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    mr = np.mean(np.max(red, axis=0))
    mg = np.mean(np.max(green, axis=0))
    mb = np.mean(np.max(blue, axis=0))
    arr = arr/np.mean([mr, mg, mb])
    arr[arr > 1] = 1
    arr = np.uint8(255*arr)
    return arr


def drowRGB(R, G, B, name):
    arr = np.zeros((len(R), len(R[0]), 3))
    coef = 0.08
    # arr[:, :, 0] = max_axis(R, k=coef)
    # arr[:, :, 1] = max_axis(G, k=coef)
    # arr[:, :, 2] = max_axis(B, k=coef)
    # drowGrafics(arr, name+'1')
    arr[:, :, 0] = R
    arr[:, :, 1] = G
    arr[:, :, 2] = B
    arr = max_axis(arr, k=coef)
    # drowGrafics(arr, name + '2')

    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    red_img = Image.fromarray(red).convert("L")
    green_img = Image.fromarray(green).convert("L")
    blue_img = Image.fromarray(blue).convert("L")
    square_img = Image.merge("RGB", (blue_img, green_img, red_img))
    square_img.save('img/' + name + '.png')


def drowGrafics(arr, name):
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.figure(figsize=(25, 10))
    print("Grafic - ", np.shape(np.arange(len(arr.flatten()))), len(arr.flatten()))
    plt.plot(np.arange(len(arr.flatten())), sorted(arr.flatten()))
    plt.savefig('img/grafics/' + name + '_graf.png', dpi=1000)


def calc_elem_cog_matr(path, name):
    mat = scipy.io.loadmat(path)
    gg_im = mat['gg_im']
    vv_im = mat['vv_im']
    gv_im = mat['gv_im']
    gg_re = mat['gg_re']
    vv_re = mat['vv_re']
    gv_re = mat['gv_re']
    gg = gg_re + gg_im * 1j
    vv = vv_re + vv_im * 1j
    gv = gv_re + gv_im * 1j
    calc_elem_t11(gg, vv, gv, 150, 3, name)
    calc_elem_t12(gg, vv, gv, 150, 3, name)
    calc_elem_t13(gg, vv, gv, 150, 3, name)
    calc_elem_t21(gg, vv, gv, 150, 3, name)
    calc_elem_t22(gg, vv, gv, 150, 3, name)
    calc_elem_t23(gg, vv, gv, 150, 3, name)
    calc_elem_t31(gg, vv, gv, 150, 3, name)
    calc_elem_t32(gg, vv, gv, 150, 3, name)
    calc_elem_t33(gg, vv, gv, 150, 3, name)


def calc_elem_t11(gg, vv, gv, ver, hor, name):
    scat_mat = np.multiply((gg+vv), np.conj((gg+vv)))
    scipy.io.savemat(name + '_cog_matr/t11.mat', {'cog_matr': compress(scat_mat, ver, hor)})


def calc_elem_t12(gg, vv, gv, ver, hor, name):
    scat_mat = np.multiply((gg+vv), np.conj((gg-vv)))
    scipy.io.savemat(name + '_cog_matr/t12.mat', {'cog_matr': compress(scat_mat, ver, hor)})


def calc_elem_t13(gg, vv, gv, ver, hor, name):
    scat_mat = np.multiply(2*(gg+vv), np.conj(gv))
    scipy.io.savemat(name + '_cog_matr/t13.mat', {'cog_matr': compress(scat_mat, ver, hor)})


def calc_elem_t21(gg, vv, gv, ver, hor, name):
    scat_mat = np.multiply((gg-vv), np.conj((gg+vv)))
    scipy.io.savemat(name + '_cog_matr/t21.mat', {'cog_matr': compress(scat_mat, ver, hor)})


def calc_elem_t22(gg, vv, gv, ver, hor, name):
    scat_mat = np.multiply((gg-vv), np.conj((gg-vv)))
    scipy.io.savemat(name + '_cog_matr/t22.mat', {'cog_matr': compress(scat_mat, ver, hor)})


def calc_elem_t23(gg, vv, gv, ver, hor, name):
    scat_mat = np.multiply(2*(gg-vv), np.conj(gv))
    scipy.io.savemat(name + '_cog_matr/t23.mat', {'cog_matr': compress(scat_mat, ver, hor)})


def calc_elem_t31(gg, vv, gv, ver, hor, name):
    scat_mat = np.multiply(2*(gg+vv), np.conj(gv))
    scipy.io.savemat(name + '_cog_matr/t31.mat', {'cog_matr': compress(scat_mat, ver, hor)})


def calc_elem_t32(gg, vv, gv, ver, hor, name):
    scat_mat = np.multiply(2*(gg-vv), np.conj(gv))
    scipy.io.savemat(name + '_cog_matr/t32.mat', {'cog_matr': compress(scat_mat, ver, hor)})


def calc_elem_t33(gg, vv, gv, ver, hor, name):
    scat_mat = np.multiply(4*gv, np.conj(gv))
    scipy.io.savemat(name + '_cog_matr/t33.mat', {'cog_matr': compress(scat_mat, ver, hor)})


def cogerent(path, name):
    mat11 = scipy.io.loadmat(path + '/t11.mat')
    mat12 = scipy.io.loadmat(path + '/t12.mat')
    mat13 = scipy.io.loadmat(path + '/t13.mat')
    mat21 = scipy.io.loadmat(path + '/t21.mat')
    mat22 = scipy.io.loadmat(path + '/t22.mat')
    mat23 = scipy.io.loadmat(path + '/t23.mat')
    mat31 = scipy.io.loadmat(path + '/t31.mat')
    mat32 = scipy.io.loadmat(path + '/t32.mat')
    mat33 = scipy.io.loadmat(path + '/t33.mat')
    t11 = mat11['cog_matr']
    t12 = mat12['cog_matr']
    t13 = mat13['cog_matr']
    t21 = mat21['cog_matr']
    t22 = mat22['cog_matr']
    t23 = mat23['cog_matr']
    t31 = mat31['cog_matr']
    t32 = mat32['cog_matr']
    t33 = mat33['cog_matr']
    scat_mat = np.zeros((len(t33), len(t33[0]), 3, 3))
    scat_mat = scat_mat.astype(dtype=np.complex64, copy=False)
    for i in range(len(scat_mat)):
        for j in range(len(scat_mat[0])):
            scat_mat[i][j] = 0.5*np.array([[t11[i][j], t12[i][j], t13[i][j]],
                                           [t21[i][j], t22[i][j], t23[i][j]],
                                           [t31[i][j], t32[i][j], t33[i][j]]])
    print(np.shape(scat_mat))
    scipy.io.savemat(name + '_cog_matr/scat_mat.mat', {'scat_mat': scat_mat})

def decompositionMatrix(path, name):
    mat = scipy.io.loadmat(path)
    gg_im = mat['gg_im']
    vv_im = mat['vv_im']
    vg_im = mat['vg_im']
    gv_im = mat['gv_im']
    gg_re = mat['gg_re']
    vv_re = mat['vv_re']
    vg_re = mat['vg_re']
    gv_re = mat['gv_re']

    s_gg = gg_re + gg_im*1j
    s_vv = vv_re + vv_im*1j
    s_vg = vg_re + vg_im*1j
    s_gv = gv_re + gv_im*1j

    scat_mat = createScatterMatrix(s_gg, s_vv, s_vg, s_gv)
    alpha = gg_re
    betta = vv_re
    gamma = vg_re
    cof = 2**(1/2)

    for i in range(len(gg_re)):
        for j in range(len(gg_re[0])):
            alpha[i][j] = abs((scat_mat[i][j][1][1] + scat_mat[i][j][0][0])/cof)
            betta[i][j] = abs((scat_mat[i][j][0][0] - scat_mat[i][j][1][1])/cof)
            gamma[i][j] = abs(scat_mat[i][j][0][1] * cof)

    alpha = alpha ** 2
    betta = betta ** 2
    gamma = gamma ** 2
    drowRGB(alpha, betta, gamma, name)


def createScatterMatrix(s_gg, s_vv, s_vg, s_gv):
    scat_mat = [[[[0, 1], [2, 3]]] * len(s_gg[0])] * len(s_gg)
    scat_mat = np.array(scat_mat)
    scat_mat = scat_mat.astype(dtype=complex, copy=False)
    for i in range(len(scat_mat)):
        for j in range(len(scat_mat[0])):
            scat_mat[i][j][0][0] = s_gg[i][j]
            scat_mat[i][j][0][1] = s_vg[i][j]
            scat_mat[i][j][1][0] = s_gv[i][j]
            scat_mat[i][j][1][1] = s_vv[i][j]
    return scat_mat


def H_alpha(path, name):
    mat = scipy.io.loadmat(path)
    cog_matr = mat['scat_mat']
    alpha = np.zeros((len(cog_matr), len(cog_matr[0])))
    alpha = alpha.astype(dtype=complex, copy=False)

    h = np.zeros((len(cog_matr), len(cog_matr[0])))
    h = h.astype(dtype=complex, copy=False)
    for i in range(len(h)):
        for j in range(len(h[0])):
            d, v = np.linalg.eig(cog_matr[i][j])
            a = d[0]
            b = d[1]
            c = d[2]
            summ = a + b + c
            pk1 = np.absolute(a / summ)
            pk2 = np.absolute(b / summ)
            pk3 = np.absolute(c / summ)
            h[i][j] = -(pk1*np.math.log(pk1, 3) + pk2*np.math.log(pk2, 3) + pk3*np.math.log(pk3, 3))
            alpha_a = np.arccos(v[0][0] * np.exp(-1j * np.angle(v[0][0])))
            alpha_b = np.arccos(v[0][1] * np.exp(-1j * np.angle(v[0][1])))
            alpha_c = np.arccos(v[0][2] * np.exp(-1j * np.angle(v[0][2])))
            alpha[i][j] = (alpha_a * pk1 + alpha_b * pk2 + alpha_c * pk3) * (180 / np.pi)

    fig, ax = plt.subplots()

    arr = np.zeros((len(h), len(h[0]), 3))
    arr[:, :, 0] = np.absolute(cog_matr[:, :, 2, 2])
    arr[:, :, 1] = np.absolute(cog_matr[:, :, 1, 1])
    arr[:, :, 2] = np.absolute(cog_matr[:, :, 0, 0])
    color_cog = rescale(norm_3_4(np.absolute(arr)))

    print("ready plt")
    color_cog = color_cog.reshape(len(color_cog) * len(color_cog[0]), 3)

    # grafics_H_alpha(color_cog)
    # import sys
    # sys.exit(0)
    ax.scatter(np.absolute(h), np.absolute(alpha), marker=".", c=color_cog)

    print("finish plt")

    #  Устанавливаем интервал основных и вспомогательных делений:
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(2))
    ax.tick_params(axis='both',  # Применяем параметры к обеим осям
                   which='major',  # Применяем параметры к основным делениям
                   direction='in',  # Рисуем деления внутри и снаружи графика
                   length=10,  # Длинна делений
                   width=2,  # Ширина делений
                   color='black',  # Цвет делений
                   pad=5,  # Расстояние между черточкой и ее подписью
                   labelsize=20,  # Размер подписи
                   labelcolor='black',  # Цвет подписи
                   bottom=True,  # Рисуем метки снизу
                   top=False,  # сверху
                   left=True,  # слева
                   right=False,  # и справа
                   labelbottom=True,  # Рисуем подписи снизу
                   labeltop=False,  # сверху
                   labelleft=True,  # слева
                   labelright=False,  # и справа
                   labelrotation=0)  # Поворот подписей

    #  Настраиваем вид вспомогательных тиков:
    ax.tick_params(axis='both',  # Применяем параметры к обеим осям
                   which='minor',  # Применяем параметры к вспомогательным делениям
                   direction='in',  # Рисуем деления внутри и снаружи графика
                   length=5,  # Длинна делений
                   width=1,  # Ширина делений
                   color='black',  # Цвет делений
                   pad=5,  # Расстояние между черточкой и ее подписью
                   labelsize=20,  # Размер подписи
                   labelcolor='black',  # Цвет подписи
                   bottom=True,  # Рисуем метки снизу
                   top=False,  # сверху
                   left=True,  # слева
                   right=False)  # и справа
    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 90])
    #  Добавляем линии основной сетки:
    ax.grid(which='major', color='gray')
    fig.set_figwidth(18)
    fig.set_figheight(14)
    #  Включаем видимость вспомогательных делений:
    ax.minorticks_on()
    plt.savefig(name + '_Entropy_diagramm.png', dpi=1000)


def compress(original, ver, hor):
    current_matrix = np.zeros((math.ceil(len(original) / ver), math.ceil(len(original[0]) / hor)))
    current_matrix = current_matrix.astype(dtype=np.complex64, copy=False)
    for row1 in range(0, len(original), ver):
        for row2 in range(0, len(original[0]), hor):
            current_matrix[math.trunc(row1 / ver)][math.trunc(row2 / hor)] = np.sum(
                original[row1:row1 + ver, row2:row2 + hor])
    return current_matrix

def grafics_H_alpha(cog):

    x = np.arange(len(cog[:, :, 2].flatten()))
    fig, axes = plt.subplots(2, 2)


    axes[0][0].scatter(x, sorted(cog[:, :, 0].flatten()))
    axes[0][0].set_title('Цвет каждой точки по строке-ключу')

    axes[0][1].scatter(x, sorted(cog[:, :, 1].flatten()))
    axes[0][1].set_title('Цвет каждой точки из встроенной палитры')

    axes[1][0].scatter(x, sorted(cog[:, :, 2].flatten()))
    axes[1][0].set_title('RGB цвет каждой точки')

    fig.set_figwidth(22)    #  ширина и
    fig.set_figheight(12)    #  высота "Figure"

    plt.show()


if __name__ == '__main__':
    # calc_elem_cog_matr('file/led_data.mat','led')
    # cogerent('led_cog_matr','led')
    # H_alpha('led_cog_matr/scat_mat.mat','led_RGB')
    # H_alpha('led_cog_matr/scat_mat.mat', 'led_RGB')

    decompositionMatrix('pos.mat', 'pos_norm_max_axis')
    # path = 'pos_simple.mat'
    # name = 'pos_med4'
    # mat = scipy.io.loadmat(path)
    # alpha = mat['alpha']
    # betta = mat['betta']
    # gamma = mat['gamma']
    # drowRGB(alpha, betta, gamma, name)

# profiler
# numba jit



