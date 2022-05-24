import sys

def separate_dem_section(filename, out, x, y, w, h):
    from osgeo import gdal

    # abrir camada de entrada
    src_f = filename
    src_ds = gdal.Open(src_f, 0)  # carrega a camada raster em um objeto "dataset"

    # definir o nome da camada de saída
    out_f = out  # incluir a extensão de saída


    #[opcional] usar a função auxiliar gdal.TranslateOptions() para configurar as opções de tradução
    #ver em https://gdal.org/python/osgeo.gdal-module.html#TranslateOptions todas as opções

    uli = x  # exemplo de coordenada I (em píxel) do canto superior esquerdo
    ulj = y  # exemplo de coordenada J (em píxel) do canto superior esquerdo
    width = w  # exemplo de largura em píxel
    height = h  # exemplo de altura em píxel

    # definição das opções:
    options = gdal.TranslateOptions(format='GTiff', bandList=[1], srcWin=[uli, ulj, width, height])

    # convocar a função Translate e passar o objeto 'options'
    out_ds = gdal.Translate(destName=out_f, srcDS=src_ds, options=options)

    # fechar os datasets:
    src_ds = None
    out_ds = None


def main():
    args = sys.argv
    filename = args[1]

    output = args[2]
    x = args[3]
    y = args[4]
    w = args[5]
    h = args[6]

    separate_dem_section(filename, output, x, y, w, h)


if __name__ == '__main__':
    main()
    # python.exe recorte.py AP_27783_FBS_F2830_RT1.dem.tif <output file name> 1750 3990 200 200