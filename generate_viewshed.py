import sys
import numpy as np
from main import Mde, r2_distance
from PIL import Image
import math


# Calcula a interceptação horizontal da LOS para um valor de x
def line_of_sight_y(p1, p2, x):
    x1, y1 = p1
    x2, y2 = p2
    return x * ((y2-y1)/(x2-x1)) + (x2*y1 - x1*y2)/(x2-x1)


# Calcula a interceptação vertical da LOS para um valor de y
def line_of_sight_x(p1, p2, y):
    x1, y1 = p1
    x2, y2 = p2
    return x1 + (y - y1) * (x2 - x1) / (y2 - y1)


def calculate_slope(x1, x2, y1, y2, z1, z2):
    return (z2 - z1) / r2_distance(x1, x2, y1, y2)


# Retorna as coordenadas x, y do canto superior esquerdo da célula
def cell_coordinates(i, j, cell_size):
    return j * cell_size, i * cell_size


# Retorna 1 se a célula alvo for visível do ponto O, caso contrário retorna 0
def is_visible(grid, o_i, o_j, t_i, t_j, cell_size):

    o_x, o_y = cell_coordinates(o_i, o_j, cell_size)
    t_x, t_y = cell_coordinates(t_i, t_j, cell_size)

    o_t_slope = calculate_slope(o_x, t_x, o_y, t_y, grid[o_i, o_j] +observer_height, grid[t_i, t_j] +5)

    # 3 casos:
    # Observador e alvo na mesma linha ou coluna
    if o_i == t_i or o_j == t_j:
        # Lista de células p intermediárias
        center_vector = list()

        if o_i > t_i:
            for p_i in range(t_i + 1, o_i):
                center_vector.append((o_j, p_i))
        elif t_i > o_i:
            for p_i in range(o_i + 1, t_i):
                center_vector.append((o_j, p_i))
        elif o_j > t_j:
            for p_j in range(t_j + 1, o_j):
                center_vector.append((p_j, o_i))
        elif t_j > o_j:
            for p_j in range(o_j + 1, t_j):
                center_vector.append((p_j, o_i))

        # Verifica se algum dos pontos intermediários obstrui a visão do observador,
        # ou seja, se a inclinação entre O e P é maior que a inclinação entre O e T
        for p_j, p_i in center_vector:
            p_x, p_y = cell_coordinates(p_i, p_j, cell_size)
            o_p_slope = calculate_slope(o_x, p_x, o_y, p_y, grid[o_i, o_j] +observer_height, grid[p_i, p_j])
            if o_p_slope > o_t_slope:
                return 0
        return 1

    # Observador e alvo com 1 linha OU 1 coluna de diferença
    elif abs(o_j - t_j) == 1 or abs(o_i - t_i) == 1:
        left_vector = list()
        right_vector = list()
        if (o_j == t_j + 1 or o_j == t_j - 1) and o_i < t_i:
            for p_i in range(o_i, t_i+1):
                left_vector.append((t_j, p_i))
            for p_i in range(o_i +1, t_i):
                right_vector.append((o_j, p_i))

        elif (o_j == t_j + 1 or o_j == t_j - 1) and o_i > t_i:
            for p_i in range(t_i, o_i+1):
                left_vector.append((t_j, p_i))
            for p_i in range(t_i +1, o_i):
                right_vector.append((o_j, p_i))

        elif (o_i == t_i + 1 or o_i == t_i -1) and o_j < t_j:
            for p_j in range(o_j, t_j):
                left_vector.append((p_j, t_i))
            for p_j in range(o_j +1, t_j +1):
                right_vector.append((p_j, o_i))

        elif (o_i == t_i + 1 or o_i == t_i -1) and o_j > t_j:
            for p_j in range(t_j, o_j+1):
                left_vector.append((p_j, t_i))
            for p_j in range(t_j +1, o_j):
                right_vector.append((p_j, o_i))

        left_blocked = 0
        right_blocked = 0
        # Verifica se algum dos pontos intermediários do vetor esquerdo obstrui a visão do observador,
        # ou seja, se a inclinação entre O e P é maior que a inclinação entre O e T
        for p_j, p_i in left_vector:
            p_x, p_y = cell_coordinates(p_i, p_j, cell_size)
            o_p_slope = calculate_slope(o_x, p_x, o_y, p_y, grid[o_i, o_j] +observer_height, grid[p_i, p_j])
            if o_p_slope > o_t_slope:
                left_blocked = 1
                break
        # Verifica se algum dos pontos intermediários do vetor direito obstrui a visão do observador,
        # ou seja, se a inclinação entre O e P é maior que a inclinação entre O e T
        for p_j, p_i in right_vector:
            p_x, p_y = cell_coordinates(p_i, p_j, cell_size)
            o_p_slope = calculate_slope(o_x, p_x, o_y, p_y, grid[o_i, o_j] +observer_height, grid[p_i, p_j])
            if o_p_slope > o_t_slope:
                right_blocked = 1
                break

        # Se existir bloqueio nos dois vetores a célula é considerada não visível,
        # caso contrário é visível
        if right_blocked and left_blocked:
            return 0
        return 1

    # Todos os outros casos
    else:
        left_vector = list()
        right_vector = list()
        # Definir os cantos do left vector e right vector
        # \\
        if (t_i > o_i and t_j > o_j) or (t_i < o_i and t_j < o_j):
            l1 = (o_j+1, o_i)
            l2 = (t_j+1, t_i)
            l3 = (o_j, o_i+1)
            l4 = (t_j, t_i+1)
        # //
        else:
            l1 = (o_j, o_i)
            l2 = (t_j, t_i)
            l3 = (o_j +1, o_i +1)
            l4 = (t_j +1, t_i +1)

        # Busca os pontos de interceptação horizontal
        if l1[1] < l2[1]:
            i1 = l1[1]
            i2 = l2[1]
            i3 = l3[1]
            i4 = l4[1]
        else:
            i1 = l2[1]
            i2 = l1[1]
            i3 = l4[1]
            i4 = l3[1]


        # Itera pelos cortes horizontas em Y, nas fronteiras das células
        for i in range(i1 +1, i2):
            p_i1 = i
            p_i2 = i - 1
            # Calcula a coluna interceptada na reta no ponto de y = i
            p_j = math.floor(line_of_sight_x(l1, l2, i))
            left_vector.append((p_j,p_i1))
            left_vector.append((p_j,p_i2))

        for i in range(i3 +1, i4):
            p_i1 = i
            p_i2 = i - 1
            # Calcula a coluna interceptada na reta no ponto de y = i
            p_j = math.floor(line_of_sight_x(l3, l4, i))
            right_vector.append((p_j,p_i1))
            right_vector.append((p_j,p_i2))

        # Busca os pontos de interceptação vertical
        if l1[0] < l2[0]:
            j1 = l1[0]
            j2 = l2[0]
            j3 = l3[0]
            j4 = l4[0]
        else:
            j1 = l2[0]
            j2 = l1[0]
            j3 = l4[0]
            j4 = l3[0]
        # Itera pelos cortes verticais em X, nas fronteiras das células
        for j in range(j1 + 1, j2):
            p_j1 = j
            p_j2 = j - 1
            # Calcula a linha interceptada na reta no ponto de x = j
            p_i = math.floor(line_of_sight_y(l1, l2, j))
            left_vector.append((p_j1, p_i))
            left_vector.append((p_j2, p_i))

        for j in range(j3 + 1, j4):
            p_j1 = j
            p_j2 = j - 1
            # Calcula a linha interceptada na reta no ponto de x = j
            p_i = math.floor(line_of_sight_y(l3, l4, j))
            right_vector.append((p_j1, p_i))
            right_vector.append((p_j2, p_i))

        left_vector = list(dict.fromkeys(left_vector))
        right_vector = list(dict.fromkeys(right_vector))

        left_blocked = 0
        right_blocked = 0

        # Verifica se algum dos pontos intermediários do vetor esquerdo obstrui a visão do observador,
        # ou seja, se a inclinação entre O e P é maior que a inclinação entre O e T
        for p_j, p_i in left_vector:
            p_x, p_y = cell_coordinates(p_i, p_j, cell_size)
            o_p_slope = calculate_slope(o_x, p_x, o_y, p_y, grid[o_i, o_j] +observer_height, grid[p_i, p_j])
            if o_p_slope > o_t_slope:
                left_blocked = 1
                break

        # Verifica se algum dos pontos intermediários do vetor direito obstrui a visão do observador,
        # ou seja, se a inclinação entre O e P é maior que a inclinação entre O e T
        for p_j, p_i in right_vector:
            p_x, p_y = cell_coordinates(p_i, p_j, cell_size)
            o_p_slope = calculate_slope(o_x, p_x, o_y, p_y, grid[o_i, o_j] +observer_height, grid[p_i, p_j])
            if o_p_slope > o_t_slope:
                right_blocked = 1
                break

        if left_blocked and right_blocked:
            return 0
        return 1


# Função que recebe a matriz de elevação, a posição do observador,
# o alcance do campo de visão e a largura das células do grid e retorna o viewshed
def generate_viewshed(grid, o_i, o_j, fov_radius, cell_size, o_h):
    global observer_height
    observer_height = o_h
    rows, cols = grid.shape

    # Variável que armazena a visibilidade de cada pixel do mapa com valores de [0,1]
    viewshed = np.zeros((rows, cols))

    # Itera todas as células do grid, t_i = linha do "target", t_j = coluna do "target"
    # e verifica se está oculta ou não
    for t_i in range(rows):
        for t_j in range(cols):
            # Célula do grid está fora de alcance do observador

            dist = r2_distance(t_j, o_j, t_i, o_i)

            # Célula alvo é a própria célula do observador -> totalmente visível
            if t_j == o_j and t_i == o_i:
                viewshed[t_i, t_j] = 1
                continue

            if t_j == o_j and t_i == o_i:
                print('obs')

            # Célula alvo fora do alcance de visão do observador
            elif dist > fov_radius:
                continue

            else:
                viewshed[t_i, t_j] = is_visible(grid, o_i, o_j, t_i, t_j, cell_size)
                if viewshed[t_i, t_j]:
                    viewshed[t_i, t_j] = (fov_radius - dist) / fov_radius

    return viewshed


def save_viewshed_image(viewshed, output_file):
    im = Image.fromarray(viewshed * 255).convert('RGB')
    im.save(output_file)


def main():
    args = sys.argv
    filename = args[1]

    output_file = args[2]

    col = int(args[3])
    row = int(args[4])
    radius = float(args[5])
    observer_height = int(args[6])
    reduction_factor = int(args[7])

    mde = Mde(filename, reduction_factor)

    viewshed = generate_viewshed(mde.grid, row, col, radius, mde.get_cell_size(), observer_height)

    save_viewshed_image(viewshed, output_file)

if __name__ == '__main__':
    main()
    # python.exe generate_viewshed.py <input file name> <output file name> <observer_col> <observer_row> <radius>