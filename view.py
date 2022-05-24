from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cbook import get_sample_data
from pylab import rcParams
#from matplotlib._png import read_png
import numpy as np
#import imageio
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from main import Mde
from PIL import Image
import sys

#Recebe DEM e salva em PNG
def tif_to_png(mde, f_out):
  im = Image.fromarray(mde).convert('RGB')
  im.save(f_out)

def visited_opened_nodes(img2, opened, visited):
    img = img2 * 0 + 0.5
    #for i, j in opened:
    #    img[i, j] = [0.8, 0.8, 0.2]
    for i, j in visited:
        img[i, j] = [0.6, 0.3, 0.6]
    return img

def define_tom_viewshed(img2):
    fundo = 0.7
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if sum(img2[i,j,:]) == 0:
                img2[i,j,:] = [fundo,fundo,fundo]
            else:
                img2[i,j,0] = ((1 - fundo)*img2[i,j,0]) + fundo
                #img2[i,j,1:] = np.sqrt((img2[i,j,1:] -0.8)**2)
                img2[i, j, 1] = ((0 - fundo)*img2[i,j,1]) + fundo
                img2[i, j, 2] = ((0 - fundo) * img2[i, j, 2]) + fundo
    return img2


def colorbar():
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    import matplotlib.colors

    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    xdata = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    ydata = [x * x for x in xdata]
    norm = plt.Normalize(1, 150)
    colorlist = [(0.7,0.7,0.7), (1,0,0)]
    newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)

    cmap = newcmp
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, orientation='horizontal',
                 label="Visibilidade")


# Desenha a superfície a partir do PNG do terreno
def draw_surface(file, projection, alt_diff):
    plt.rcParams["figure.autolayout"] = False
    img = mpimg.imread(file)
    z = img[:, :, :-1] * alt_diff

    #opened = open_csv('opened.csv')
    #visited = open_csv('visited.csv')

    img2 = mpimg.imread(projection)
    img2 = define_tom_viewshed(img2)

    img2 = img

    #img2 = visited_opened_nodes(img2, opened, visited)

    x, y = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    fig = plt.figure(figsize=(20, 20), dpi=80)
    ax = plt.axes(projection='3d', computed_zorder=False, proj_type='persp')

    px, py = np.mgrid[0:10, 0:10]

    px = px * ((img.shape[0] - 1) / 10) + 10
    py = py * ((img.shape[1] - 1) / 10) + 10

    px = px.astype(int)
    py = py.astype(int)

    ax.plot_surface(x, y, z[x, y, 0], rstride=1, cstride=1,
                    facecolors=img2, zorder=4.4)

    viewpoints = [(145, 173), (156, 121), (156, 71), (59, 172), (90, 117), (103, 26), (15, 121), (52, 91), (45, 17)]
    #viewpoints = []
    i = 0
    for vp_y, vp_x in viewpoints:
        i = i+1
        mk = '$' + str(i) + '$'
        ax.scatter(vp_y, vp_x, z[vp_y, vp_x, 0], marker='o', color='w', zorder=4.55, s=300)
        ax.scatter(vp_y, vp_x, z[vp_y, vp_x, 0], marker=mk, color='r', zorder=4.6, s=200)

    #path = open_csv('path.csv')
    path = []
    # Pontos do caminho
    for cell in path:
        ax.scatter(cell[0], cell[1], z[cell[0], cell[1], 0], marker='o', c='c', zorder=4.5, s=10)

    #ax.scatter(100, 100, z[100, 100, 0], marker='o', c='b', zorder=4.6)

    #ax.view_init(90, 180)
    ax.view_init(75, 135)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    ax.grid(False)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.xlabel('X', fontsize=16, labelpad=16)
    plt.ylabel('Y', fontsize=16, labelpad=16)

    # Set general font size
    #plt.rcParams['font.size'] = '16'

    # Set tick font size
    for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        label.set_fontsize(16)

    #colorbar()

    plt.show()

from csv import reader

def open_csv(f):
  with open(f, 'r') as read_obj:
      list_of_rows = []
      csv_reader = reader(read_obj)
      for l in csv_reader:
        if len(l) == 0:
          continue
        l = list(map(int, l))
        list_of_rows.append(l)
      return list_of_rows


def count_lines(f):
    n_lines = 0
    with open(f, 'r') as read_obj:
        csv_reader = reader(read_obj)
        for _ in csv_reader:
            n_lines = n_lines + 1
    return n_lines

def main():
    #print(count_lines('dataset.csv'))
    #return
    args = sys.argv

    filename = args[1]
    viewshed = './VIEWSHEDS/todos.png'

    #Dimensão em pixels da area do nodo = reduction_factor X reduction_factor
    reduction_factor = int(args[2])
    mde = Mde(filename, reduction_factor)
    view = Mde(viewshed, 1)

    max_alt = mde.grid.max()
    min_alt = mde.grid.min()
    diff = max_alt - min_alt

    mde.grid = (mde.grid - min_alt) * (255/diff)

    tif_to_png(np.array(mde.grid), 'terrain.png')
    #tif_to_png(np.array(view.grid), 'viewshed.png')
    draw_surface('terrain.png', viewshed, diff)
    #draw_surface('terrain.png', 'visibility.png', diff)

if __name__ == '__main__':
    main()