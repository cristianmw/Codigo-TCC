import sys
from osgeo import gdal
import numpy as np


def main():
    args = sys.argv
    filename = args[1]

    src_f = filename
    src_ds = gdal.Open(src_f, 0)
    band = src_ds.GetRasterBand(1)
    ulx, xres, xskew, uly, yskew, yres = src_ds.GetGeoTransform()
    output = args[2]

    x = float(args[3])
    y = float(args[4])
    dist = float(args[5])

    x_coord = ulx + x * xres
    y_coord = uly + y * yres
    dist_translated = dist * xres


    gdal.ViewshedGenerate(srcBand=band, driverName='GTiff', targetRasterName=output, creationOptions=None,
                          observerX=x_coord, observerY=y_coord, observerHeight=10, targetHeight=5, visibleVal=255,
                          invisibleVal=0, outOfRangeVal=0, noDataVal=0, dfCurvCoeff=0.85714, mode=2,
                          maxDistance=dist_translated)



if __name__ == '__main__':
    main()
    # python.exe viewshed.py dem.tif <output file name> 1750 3990 200 200