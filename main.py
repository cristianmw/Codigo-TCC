import io
import shutil
import random
import sys
import numpy as np
import math
import csv
import heapq
import os
import glob
import matplotlib.image as mpimg
from time import process_time
from numba import cuda, jit
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

def r2_distance(x1, x2, y1, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def r3_distance(x1, x2, y1, y2, z1, z2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


class Mde:
    # https://rasterio.readthedocs.io/en/latest/quickstart.html
    import rasterio

    def __init__(self, fp, reduction_factor):
        self.dataset = self.rasterio.open(fp)
        self.band1 = self.dataset.read(1)
        self.pixel_resolution = self.dataset.transform[0]
        self.h_limit = self.dataset.height
        self.w_limit = self.dataset.width
        self.generate_grid(reduction_factor)
        self.cell_size = self.pixel_resolution * reduction_factor
        global CELL_HEIGHT, CELL_WIDTH, GRID_ROWS, GRID_COLS, GRID_HEIGHT, GRID_WIDTH
        CELL_HEIGHT = self.pixel_resolution * reduction_factor
        CELL_WIDTH = self.pixel_resolution * reduction_factor
        GRID_COLS = self.grid.shape[0]
        GRID_ROWS = self.grid.shape[1]
        GRID_WIDTH = CELL_WIDTH * GRID_COLS
        GRID_HEIGHT = CELL_HEIGHT * GRID_ROWS

    def generate_grid(self, reduction_factor):
        x = int(self.h_limit / reduction_factor)
        y = int(self.w_limit / reduction_factor)
        self.grid = np.zeros(shape=(x, y))
        for i in range(x):
            for j in range(y):
                sub_section = self.band1[i * reduction_factor: (i + 1) * reduction_factor, j * reduction_factor: (j + 1) * reduction_factor]
                self.grid[i, j] = np.sum(sub_section)
                self.grid[i, j] = round(self.grid[i, j] / (len(sub_section) * len(sub_section[0])))

    def get_cell_size(self):
        return self.cell_size

    def get_grid_width(self):
        return GRID_WIDTH

    def get_grid_height(self):
        return GRID_HEIGHT

class Vertex:
    def __init__(self, elevation, node_id):
        self.local_risk = 0
        self.elevation = elevation
        self.id = node_id
        self.edges = {}
        self.distance = 99999999
        self.risk = 99999999
        self.previous = None
        self.visited = False

    def __str__(self):
        return str(self.id) + ' elevation: ' + str(self.elevation) + ' coord: ' + str(self.get_r2_coordinates()) + ' edges: ' + str([x for x in self.edges.keys()]) + str([x for x in self.edges.values()])

    def add_edge(self, node_id, edge_weight):
        self.edges[node_id] = edge_weight

    def get_id(self):
        return self.id

    def get_edges(self):
        return self.edges

    def get_neighbors(self):
        return self.edges.keys()

    def get_x(self):
        return self.get_j() * CELL_WIDTH

    def get_y(self):
        return self.get_i() * CELL_HEIGHT

    def get_i(self):
        return math.floor(self.id / GRID_ROWS)

    def get_j(self):
        return self.id % GRID_COLS

    def get_r2_coordinates(self):
        return self.get_x(), self.get_y()

    def get_coordinates(self):
        return self.get_i(), self.get_j()

    def get_elevation(self):
        return self.elevation

    def get_edge_weight(self, vertex_id):
        return self.edges[vertex_id]

    def set_previous(self, prev):
        self.previous = prev

    def set_visited(self, visit):
        self.visited = visit

    def set_distance(self, distance):
        self.distance = distance

    def get_distance(self):
        return self.distance

    def set_risk(self, risk):
        self.risk = risk

    def get_risk(self):
        return self.risk

    def reset(self):
        self.distance = 99999999
        self.risk = 99999999
        self.previous = None
        self.visited = False

    def set_local_risk(self, local_risk):
        self.local_risk = local_risk

    def get_local_risk(self):
        return self.local_risk

    def __lt__(self, other):
        return self.distance < other.distance

    def __eq__(self, other):
        return self.id == other.get_id()

class Graph:
    def __init__(self, mde):
        self.vertices = {}
        self.max_edge = 0.0
        self.min_edge = float("inf")
        self.create_vertices(mde)
        self.generate_edges(False)

    def __iter__(self):
        return iter(self.vertices.values())

    def reset(self):
        for v in self:
            v.reset()

    def __str__(self):
        for v in self:
            print(v)

    def create_vertices(self, mde):
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                vertex_elevation = mde.grid[i,j]
                vertex_id = i * GRID_COLS + j
                self.add_vertex(vertex_elevation, vertex_id)

    def update_vertices_risk(self, viewshed):
        for v in self:
            i,j = v.get_coordinates()
            v.set_local_risk(viewshed[i,j])

    def get_vertex(self, id):
        if id in self.vertices:
            return self.vertices[id]
        else:
            return None

    def get_vertex_by_coords(self, i, j):
        id = get_id_by_coords(i,j)
        return self.get_vertex(id)

    def get_vertices(self):
        return self.vertices.keys()

    def add_vertex(self, elevation, id):
        self.vertices[id] = Vertex(elevation, id)

    def generate_edges(self, diagonal):
        for vertex_id, vertex in self.vertices.items():
            vertex_id = vertex.get_id()
            i, j = vertex.get_coordinates()

            j1 = j + 1
            i1 = i
            if j1 < GRID_COLS:
                vertex2_id = i1 * GRID_COLS + j1
                vertex2 = self.get_vertex(vertex2_id)
                if vertex2:
                    weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),
                                         vertex.get_elevation(), vertex2.get_elevation())
                    if weight > self.max_edge:
                        self.max_edge = weight
                    if weight < self.min_edge:
                        self.min_edge = weight
                    vertex.add_edge(vertex2_id, weight)


            j1 = j - 1
            i1 = i
            if j1 >= 0:
                vertex2_id = i1 * GRID_COLS + j1
                vertex2 = self.get_vertex(vertex2_id)
                if vertex2:
                    weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),
                                           vertex.get_elevation(), vertex2.get_elevation())
                    if weight > self.max_edge:
                        self.max_edge = weight
                    if weight < self.min_edge:
                        self.min_edge = weight
                    vertex.add_edge(vertex2_id, weight)


            j1 = j
            i1 = i + 1
            if i1 < GRID_ROWS:
                vertex2_id = i1 * GRID_COLS + j1
                vertex2 = self.get_vertex(vertex2_id)
                if vertex2:
                    weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),
                                           vertex.get_elevation(), vertex2.get_elevation())
                    if weight > self.max_edge:
                        self.max_edge = weight
                    if weight < self.min_edge:
                        self.min_edge = weight
                    vertex.add_edge(vertex2_id, weight)

            j1 = j
            i1 = i - 1
            if i1 >= 0:
                vertex2_id = i1 * GRID_COLS + j1
                vertex2 = self.get_vertex(vertex2_id)
                if vertex2:
                    weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),
                                           vertex.get_elevation(), vertex2.get_elevation())
                    if weight > self.max_edge:
                        self.max_edge = weight
                    if weight < self.min_edge:
                        self.min_edge = weight
                    vertex.add_edge(vertex2_id, weight)

            if diagonal:
                j1 = j + 1
                i1 = i + 1
                if j1 < GRID_COLS and i1 < GRID_ROWS:
                    vertex2_id = i1 * GRID_COLS + j1
                    vertex2 = self.get_vertex(vertex2_id)
                    if vertex2:
                        weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),
                                             vertex.get_elevation(), vertex2.get_elevation())
                        if weight > self.max_edge:
                            self.max_edge = weight
                        if weight < self.min_edge:
                            self.min_edge = weight
                        vertex.add_edge(vertex2_id, weight)

                j1 = j - 1
                i1 = i + 1
                if j1 >= 0 and i1 < GRID_ROWS:
                    vertex2_id = i1 * GRID_COLS + j1
                    vertex2 = self.get_vertex(vertex2_id)
                    if vertex2:
                        weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),
                                             vertex.get_elevation(), vertex2.get_elevation())
                        if weight > self.max_edge:
                            self.max_edge = weight
                        if weight < self.min_edge:
                            self.min_edge = weight
                        vertex.add_edge(vertex2_id, weight)

                j1 = j + 1
                i1 = i - 1
                if i1 >= 0 and j1 < GRID_COLS:
                    vertex2_id = i1 * GRID_COLS + j1
                    vertex2 = self.get_vertex(vertex2_id)
                    if vertex2:
                        weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),
                                             vertex.get_elevation(), vertex2.get_elevation())
                        if weight > self.max_edge:
                            self.max_edge = weight
                        if weight < self.min_edge:
                            self.min_edge = weight
                        vertex.add_edge(vertex2_id, weight)

                j1 = j - 1
                i1 = i - 1
                if i1 >= 0 and j1 >= 0:
                    vertex2_id = i1 * GRID_COLS + j1
                    vertex2 = self.get_vertex(vertex2_id)
                    if vertex2:
                        weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),
                                             vertex.get_elevation(), vertex2.get_elevation())
                        if weight > self.max_edge:
                            self.max_edge = weight
                        if weight < self.min_edge:
                            self.min_edge = weight
                        vertex.add_edge(vertex2_id, weight)


    def normalize_distance(self, distance):
        return (distance - self.min_edge) / (self.max_edge - self.min_edge)

    # Ajusta os valores de visibilidade em um intervalo de [0,1] pra um intervalo [Min Slope, Max Slope]
    # Atravessar um ponto com visibilidade 0 para um ponto com visibilidade máxima equivale a percorrer
    # a inclinação máxima do mapa
    # Atravessar um ponto com mesma visibilidade equivale a menor distância entre dois nodos
    # (mesma elevação = distância no R2)
    def normalize_visibility(self, visibility):
        return visibility * (self.max_edge - self.min_edge) + self.min_edge


    # [0, max]
    def normalize_visibility2(self, visibility):
        return visibility * (self.max_edge)

    #[0, max - min]
    def normalize_visibility3(self, visibility):
        return visibility * (self.max_edge - self.min_edge)


def shortest(v, path):
    if v.previous:
        path.append(v.previous.get_id())
        shortest(v.previous, path)
    return


# Heuristica da distancia euclidiana
def r3_heuristic(start, goal):
    x1, y1 = start.get_r2_coordinates()
    x2, y2 = goal.get_r2_coordinates()
    z1 = start.get_elevation()
    z2 = goal.get_elevation()

    dst = r3_distance(x1, x2, y1, y2, z1, z2)

    return dst


# Adapted A*
def safe_astar(g, start, goal, v_weight):
    opened = []
    visited = []

    visibility_weight = v_weight

    # Set the distance for the start node to zero
    start.set_risk(start.get_local_risk())
    start.set_distance(0)

    hscore = visibility_weight * start.get_risk() + start.get_distance() + r3_heuristic(start, goal)

    # Put tuple pair into the priority queue
    unvisited_queue = [(hscore, start)]
    heapq.heapify(unvisited_queue)

    count_visited = 0
    count_open = 1

    opened.append(start.get_coordinates())

    while len(unvisited_queue):
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]

        if current == goal:
            return current.get_distance(), current.get_risk(), count_visited, count_open, opened, visited, current.get_distance() + visibility_weight * current.get_risk()

        current.set_visited(True)
        count_visited = count_visited + 1
        visited.append(current.get_coordinates())

        for next_id in current.get_neighbors():
            next = g.get_vertex(next_id)
            new_dist = current.get_distance() + current.get_edge_weight(next_id)
            new_risk = current.get_risk() + next.get_local_risk()

            if new_dist + visibility_weight * new_risk < next.get_distance() + visibility_weight * next.get_risk():
                next.set_previous(current)
                next.set_distance(new_dist)
                next.set_risk(new_risk)

                hscore = visibility_weight * new_risk + new_dist + r3_heuristic(next, goal)

                if not next.visited:
                    heapq.heappush(unvisited_queue, (hscore, next))
                    count_open = count_open + 1
                    opened.append(next.get_coordinates())


# A*
def astar(g, start, goal):
    opened = []
    visited = []

    # Set the distance for the start node to zero
    start.set_distance(0)

    hscore = start.get_distance() + r3_heuristic(start, goal)

    # Put tuple pair into the priority queue
    unvisited_queue = [(hscore, start)]
    heapq.heapify(unvisited_queue)

    count_visited = 0
    count_open = 1
    opened.append(start.get_coordinates())

    while len(unvisited_queue):
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]

        if current == goal:
            return current.get_distance(), count_visited, count_open, opened, visited

        current.set_visited(True)
        count_visited = count_visited + 1
        visited.append(current.get_coordinates())

        for next_id in current.get_neighbors():
            next = g.get_vertex(next_id)
            new_dist = current.get_distance() + current.get_edge_weight(next_id)

            if new_dist < next.get_distance():
                next.set_previous(current)
                next.set_distance(new_dist)

                hscore = new_dist + r3_heuristic(next, goal)

                if not next.visited:
                    heapq.heappush(unvisited_queue, (hscore, next))
                    count_open = count_open + 1
                    opened.append(next.get_coordinates())


def get_visited_coord(graph, visited_vertices):
    path = []
    for vertex_id in visited_vertices[::-1]:
        path.append(graph.get_vertex(vertex_id).get_coordinates())
    return path


def get_id_by_coords(i, j):
    return i * GRID_COLS + j


def save_path_csv(output, path):
    with open(output, 'w') as out:
        csv_out = csv.writer(out)
        for row in path:
            csv_out.writerow(row)


def write_dataset_csv(filename, data_io):
    with open(filename, 'a') as file:
        data_io.seek(0)
        shutil.copyfileobj(data_io, file)


def save_viewsheds(grid, viewpoints, view_radius, viewpoint_height):
    todos = np.zeros((grid.shape[0], grid.shape[1]))
    for viewpoint_i, viewpoint_j in viewpoints:
        viewshed = vs.generate_viewshed(grid, viewpoint_i, viewpoint_j, view_radius, CELL_WIDTH, viewpoint_height)
        todos = todos + viewshed
        output_file = 'VIEWSHED_' + str(viewpoint_i) + '_' + str(viewpoint_j) + '.png'
        vs.save_viewshed_image(viewshed, './VIEWSHEDS/' + output_file)
    vs.save_viewshed_image(todos, './VIEWSHEDS/todos.png')


def read_viewshed(file):
    img = mpimg.imread(file)
    viewshed = img[:, :, 0]
    return viewshed


@cuda.jit
def kernel2(M, C, U, n):
    tid = cuda.grid(1)
    if tid < n:
        if C[tid] > U[tid]:
            C[tid] = U[tid]
            M[tid] = 1
        U[tid] = C[tid]


@cuda.jit
def kernel1(V, E, W, S, M, C, U, n, b):
    tid = cuda.grid(1)
    if tid < n:
        if M[tid] == 1:
            M[tid] = 0
            start = V[tid]
            if tid == n-1:
                end = len(E)
            else:
                end = V[tid+1]
            for nid, w in zip(E[start:end], W[start:end]):
                cuda.atomic.min(U, nid, C[tid] + w + b * S[nid])


@cuda.jit
def initialize_arrays(source, n, INF, M, C, U, S, b):
    tid = cuda.grid(1)
    if tid < n:
        C[tid] = INF
        U[tid] = INF
        if tid == source:
            M[tid] = 1
            C[tid] = S[tid] * b
            U[tid] = S[tid] * b


@cuda.reduce
def sum_reduce(a, b):
    return a + b


def cuda_safe_sssp(V, E, W, S, source, b):
    n = V.shape[0]
    INF = 999999
    threadsperblock = 128
    blockspergrid = (n + (threadsperblock - 1)) // threadsperblock

    M = np.zeros(n, dtype=np.int32)
    C = np.arange(n, dtype=np.float64)
    U = np.arange(n, dtype=np.float64)
    d_M = cuda.to_device(M)
    d_C = cuda.to_device(C)
    d_U = cuda.to_device(U)
    d_V = cuda.to_device(V)
    d_E = cuda.to_device(E)
    d_W = cuda.to_device(W)
    d_S = cuda.to_device(S)

    initialize_arrays[blockspergrid, threadsperblock](source, n, INF, d_M, d_C, d_U, d_S, b)
    mask = sum_reduce(d_M)
    while mask > 0:
        kernel1[blockspergrid, threadsperblock](d_V, d_E, d_W, d_S, d_M, d_C, d_U, n, b)
        kernel2[blockspergrid, threadsperblock](d_M, d_C, d_U, n)
        mask = sum_reduce(d_M)
    C = d_C.copy_to_host()
    return C


# Cria listas de adjacências das conexões do grafo
def generate_sssp_arrays(g):
    V = []
    E = []
    W = []
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            v_id = get_id_by_coords(i, j)
            v = g.get_vertex(v_id)
            edges_index = len(E)
            V.append(edges_index)
            for u_id in v.get_neighbors():
                E.append(u_id)
                W.append(v.edges[u_id])

    return np.array(V), np.array(E), np.array(W)


def serialize_viewshed(viewshed):
    serialized_viewshed = []
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            serialized_viewshed.append(viewshed[i, j])
    return np.array(serialized_viewshed)


import generate_viewshed as vs


# Gera uma lista de tuplas de pontos de amostras para geração do dataset
def generate_sample_points(sampling_percentage):
    sections_n = 4
    sections_m = 4
    # Tamanho das divisões do mapa
    SECTION_ROWS = round(GRID_ROWS/sections_n)
    SECTION_COLS = round(GRID_COLS/sections_m)

    # Conjunto de pontos amostrados, ordenados da esquerda pra direta, de cima pra baixo
    P = []

    # Itera pelas divisões do mapa, da esquerda pra direita, de cima pra baixo
    for section_i in range(sections_n):
        for section_j in range(sections_m):
            section_points = []
            for p_i in range(SECTION_ROWS * section_i, SECTION_ROWS * section_i + SECTION_ROWS):
                for p_j in range(SECTION_COLS * section_j, SECTION_COLS * section_j + SECTION_COLS):
                    section_points.append((p_i, p_j))
            random.shuffle(section_points)
            sampling_size = int(len(section_points) * sampling_percentage)
            sample_points = section_points[0:sampling_size]
            P.extend(sample_points)
            section_points.clear()
    P.sort(key=lambda tup: (tup[1], tup[0]))
    return P


def main():
    args = sys.argv
    filename = args[1]

    reduction_factor = 2

    mde = Mde(filename, reduction_factor)

    print('Criando o grafo')
    g = Graph(mde)

    print('Gerando os pontos de amostra')
    sampling_rate = 25
    sample_coords = generate_sample_points(sampling_rate/100)
    save_path_csv("sample_points.csv", sample_coords)

    print('Gerando os viewsheds')
    #viewpoints = [(145, 173), (156, 121), (156, 71), (15, 121), (59, 172), (52, 91), (45, 17), (103, 26), (90, 117)]
    viewpoints = [(145, 173), (156, 71), (103, 26)]
    view_radius = 40
    viewpoint_height = 10

    print('Salvando os viewsheds')
    files = glob.glob('./VIEWSHEDS/*')
    for f in files:
        os.remove(f)
    save_viewsheds(mde.grid, viewpoints, view_radius, viewpoint_height)

    '''
    # ------------------ TESTE CUSTO ------------------------
    #0.0,1325.0,2056.0,3750.0,4700.0,2420.0,10939.326564426836
    visibility_map_file = './VIEWSHEDS/VIEWSHED_145_173.png'
    viewshed = read_viewshed(visibility_map_file)
    viewshed = g.normalize_visibility3(viewshed)
    g.update_vertices_risk(viewshed)
    start_id = get_id_by_coords(188,163)
    goal_id = get_id_by_coords(75,182)
    start = g.get_vertex(start_id)
    goal = g.get_vertex(goal_id)
    b = 1
    distance, risk, count_visited, count_open, opened, visited, cost = safe_astar(g, start, goal, b)
    print(distance)
    print(risk)
    print(cost)
    path = [goal.get_id()]
    shortest(goal, path)
    path2 = []
    for p_id in path:
        p = g.get_vertex(p_id)
        path2.append(p.get_coordinates())
        print(viewshed[p.get_coordinates()[0],p.get_coordinates()[1]])
    save_path_csv('path.csv', path2)
    return
    # ------------------------------------------------------
    '''

    V, E, W = generate_sssp_arrays(g)

    print('Gerando o dataset')
    for vp in viewpoints:
        start_time = process_time()
        visibility_map_file = './VIEWSHEDS/VIEWSHED_' + str(vp[0]) + '_' + str(vp[1]) + '.png'

        viewshed = read_viewshed(visibility_map_file)
        viewshed = g.normalize_visibility3(viewshed)
        S = serialize_viewshed(viewshed)

        #sample_coords = generate_sample_points(0.1)

        aux = 0
        for src_coords in sample_coords:
            data_io = io.StringIO()
            source = get_id_by_coords(src_coords[0], src_coords[1])
            b = 0.5
            C = cuda_safe_sssp(V, E, W, S, source, b)
            for dest_coords in sample_coords[aux+1:]:
                dest = get_id_by_coords(dest_coords[0], dest_coords[1])
                data_io.write("""%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n""" % (int(vp[1] * CELL_WIDTH), int(vp[0] * CELL_HEIGHT), mde.grid[vp[0], vp[1]], int(src_coords[1] * CELL_WIDTH), int(src_coords[0] * CELL_HEIGHT), mde.grid[src_coords[0], src_coords[1]], int(dest_coords[1] * CELL_WIDTH), int(dest_coords[0] * CELL_HEIGHT), mde.grid[dest_coords[0], dest_coords[1]], C[dest]))
            aux = aux +1

            write_dataset_csv('dataset_'+str(len(viewpoints))+'_'+str(sampling_rate)+'.csv', data_io)
        print('Tempo: ' + str(process_time() - start_time) + ' segundos')

    print('Dataset gerado com sucesso!')

if __name__ == '__main__':
    main()