import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.io
import fast
import numpy as np
from PIL import Image


def draw_grafics(h, alpha, color_cog, name):
    fig, ax = plt.subplots()
    ax.scatter(h, alpha, marker=".", c=color_cog)
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
    plt.savefig(name + '.png', dpi=200)

@fast.timer
def get_mat(path):
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
    return gg, vv, gv

@fast.timer
def draw_rgb(r, g, b, name, k=1):
    arr = np.zeros((len(r), len(r[0]), 3))
    arr[:, :, 0] = r
    arr[:, :, 1] = g
    arr[:, :, 2] = b
    arr = fast.max_axis(np.sqrt(np.absolute(arr)), k=k)

    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    red_img = Image.fromarray(red).convert("L")
    green_img = Image.fromarray(green).convert("L")
    blue_img = Image.fromarray(blue).convert("L")
    square_img = Image.merge("RGB", (blue_img, green_img, red_img))
    square_img.save('img/' + name + '.png')