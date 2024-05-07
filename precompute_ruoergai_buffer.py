import numpy as np
from src.utils import read_img, saveTiff

year = '2005'
# 用于制作训练可视化图
def train_visual():
    print("-----------train_visual-----------------")
    dirname = "D:\\BingqianWang\\RuoergaiClassifity\\data"
    slipts = ["clip1", "clip2"]
    for split in slipts:
        filename = "\\buffer\\training\\visual_img\\buffer_raster_" + split + ".tif"  # 有效像元栅格
        tif = read_img(dirname + filename)
        tifInfo = tif[:5]
        buffer = tif[5]
        buffer = buffer.reshape(-1, buffer.shape[2])
        # indexes = np.where((buffer == 0) | (buffer == 1)) # 有效像元筛选条件  用于筛选Buffer的样本区域
        indexes = np.where(buffer == 1)  # 有效像元筛选条件  用于整个Buffer删选
        print("Save the indexes")
        # np.save(dirname + '\\buffer\\training\\visual_img\\index_' + split + '.npy', indexes[0])
        # print("Save the bufferClassValue")
        # np.save(dirname + '\\buffer_inner_class.npy', buffer[indexes[0]])

        filenameList = [
            "\\" + year + "\\construct\\buffer\\visual_img\\" + year + "_Restoration_15_{0}.tif".format(split),
            # "\\" + year + "\\construct\\buffer\\visual_img\\EucDist_buff_inner_32bit_{0}.tif".format(split),
            # "\\" + year + "\\construct\\buffer\\visual_img\\EucDist_buff_outer_32bit_{0}.tif".format(split),
            # "\\" + year + "\\construct\\buffer\\visual_img\\Relief_500m_{0}.tif".format(split),
            # "\\" + year + "\\construct\\buffer\\visual_img\\coordinate_{0}.tif".format(split)
        ]  # 被处理栅格
        for filename in filenameList:
            tif = read_img(dirname + filename)
            tifInfo = tif[:5]
            img = tif[5]
            img = img.reshape(-1, img.shape[2])
            effectivePiex = img[indexes[0]]
            print("Save the effectivePiex")
            savename = filename.split("\\")[len(filename.split("\\")) - 1].split(".")[0]
            np.save(dirname + '\\' + year + '\\construct\\buffer\\visual_img\\' + savename + '.npy', effectivePiex)


# 用于制作Buffer的训练集与测试集
def buffer_train():
    print("-----------buffer_train-----------------")
    dirname = "D:\\BingqianWang\\RuoergaiClassifity\\data"
    slipts = [year]
    for split in slipts:
        filename = "\\" + split + "\\construct\\buffer\\mount_sample.tif"  # 有效像元栅格
        tif = read_img(dirname + filename)
        tifInfo = tif[:5]
        buffer = tif[5]
        buffer = buffer.reshape(-1, buffer.shape[2])
        indexes = np.where((buffer == 0) | (buffer == 1))  # 有效像元筛选条件  用于筛选Buffer的样本区域
        print("Save the indexes")
        np.save(dirname + '\\' + split + '\\construct\\buffer\\buffer_sample_class_index.npy', indexes[0])
        print("Save the bufferClassValue")
        np.save(dirname + '\\' + split + '\\construct\\buffer\\buffer_sample_class.npy', buffer[indexes[0]])

        # filenameList = ["\\" + split + "\\2023_Restoration_15.tif",
        #                 "\\buffer\\dis_2_buffer\\EucDist_buff_inner_32bit.tif",
        #                 "\\buffer\\dis_2_buffer\\EucDist_buff_outer_32bit.tif",
        #                 "\\buffer\\Relief_500m.tif"]  # 被处理栅格
        filenameList = ["\\" + split + "\\" + split + "_Restoration_15.tif"]  # 被处理栅格
        for filename in filenameList:
            tif = read_img(dirname + filename)
            tifInfo = tif[:5]
            img = tif[5]
            img = img.reshape(-1, img.shape[2])
            effectivePiex = img[indexes[0]]
            print("Save the effectivePiex")
            savename = filename.split("\\")[len(filename.split("\\")) - 1].split(".")[0]
            np.save(dirname + '\\' + split + '\\construct\\buffer\\' + savename + '_sample.npy', effectivePiex)


# 用于生成Buffer区域的npy（用于最后的预测）
def buffer_dataset():
    print("-----------buffer_dataset-----------------")
    dirname = "D:\\BingqianWang\\\RuoergaiClassifity\\data"
    slipts = ["buffer"]
    for split in slipts:
        filename = "\\" + split + "\\buffer_raster.tif"  # 有效像元栅格
        tif = read_img(dirname + filename)
        tifInfo = tif[:5]
        buffer = tif[5]
        buffer = buffer.reshape(-1, buffer.shape[2])
        indexes = np.where(buffer == 1)  # 有效像元筛选条件  用于整个Buffer删选
        # print("Save the indexes")
        # np.save(dirname + '\\index_' + split + '.npy', indexes[0])
        # print("Save the bufferClassValue")
        # np.save(dirname + '\\buffer_inner_class.npy', buffer[indexes[0]])

        # filenameList = ["\\2022\\2022_CDIST_sharpen.tif",
        #                 "\\buffer\\dis_2_buffer\\EucDist_buff_inner_32bit.tif",
        #                 "\\buffer\\dis_2_buffer\\EucDist_buff_outer_32bit.tif",
        #                 "\\buffer\\Relief_500m.tif"]  # 被处理栅格
        filenameList = ["\\" + year + "\\" + year + "_Restoration_15.tif"]  # 仅影像，因为两个距离图层与地形起伏度图层多年一样
        for filename in filenameList:
            tif = read_img(dirname + filename)
            tifInfo = tif[:5]
            img = tif[5]
            img = img.reshape(-1, img.shape[2])
            effectivePiex = img[indexes[0]]
            print("Save the effectivePiex")
            savename = filename.split("\\")[len(filename.split("\\")) - 1].split(".")[0]
            np.save(dirname + '\\' + year + '\\experiment\\buffer\\' + savename + '.npy', effectivePiex)


if __name__ == "__main__":
    train_visual()
    buffer_train()
    buffer_dataset()
