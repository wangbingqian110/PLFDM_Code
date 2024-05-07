from osgeo import gdal, osr
import numpy as np
from tqdm import tqdm

from src.utils import read_img, saveTiff

dirname = "D:\\BingqianWang\\\RuoergaiClassifity\\data"
referencefilename = "\\buffer\\buffer_raster.tif"

tif = read_img(dirname + referencefilename)
tifInfo = tif[:5]
height = tif[1]
width = tif[0]
num_channels = 2

# 创建一个示例的坐标图像，通道数为2（经度和纬度）
image = np.zeros((num_channels, height, width), dtype=np.float32)
# 获取地理坐标变换信息
geo_transform = tifInfo[4]

# 填充坐标图像
for i in tqdm(range(width)):
    for j in range(height):
        # 计算当前栅格的中心坐标
        x_center = geo_transform[0] + i * geo_transform[1] + 0.5 * geo_transform[1]
        y_center = geo_transform[3] + j * geo_transform[5] + 0.5 * geo_transform[5]

        # 将经度和纬度作为通道值
        image[0, j, i] = x_center
        image[1, j, i] = y_center

saveTiff(image, tifInfo[0], tifInfo[1], tifInfo[2], tifInfo[3],
         tifInfo[4], dirname + '\\core\\coordinate.tif')
