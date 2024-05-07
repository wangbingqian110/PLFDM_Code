import numpy as np
from src.utils import read_img, saveTiff

year = '2011'
Builtname = "2010_15_GHSL_build"


# 用于制作训练可视化图
def train_visual():
    print("-----------train_visual-----------------")
    dirname = "D:\\BingqianWang\\RuoergaiClassifity\\data"
    slipts = ["clip1", "clip2", 'clip3']
    for split in slipts:
        indexes = np.arange(0, 1024 * 1024)  # 有效像元筛选条件  用于整个Buffer删选
        # print("Save the indexes")
        # np.save(dirname + '\\' + year + '\\construct\\core\\visual_img\\index_' + split + '.npy', indexes)

        filenameList = [
            "\\" + year + "\\construct\\core\\visual_img\\" + year + "_Restoration_15_{0}.tif".format(split),
            "\\" + year + "\\construct\\core\\visual_img\\" + Builtname + "_{0}.tif".format(
                split)]  # 被处理栅格
        for filename in filenameList:
            tif = read_img(dirname + filename)
            tifInfo = tif[:5]
            img = tif[5]
            img = img.reshape(-1, img.shape[2])
            effectivePiex = img
            print("Save the effectivePiex")
            savename = filename.split("\\")[len(filename.split("\\")) - 1].split(".")[0]
            np.save(dirname + '\\' + year + '\\construct\\core\\visual_img\\' + savename + '.npy', effectivePiex)


# 用于制作Core的训练集与测试集
def core_train():
    print("-----------core_train-----------------")
    dirname = "D:\\BingqianWang\\RuoergaiClassifity\\data"
    slipts = [year]
    for split in slipts:
        filename = "\\" + split + "\\construct\\core\\sample_raster_" + split + ".tif"  # 有效像元栅格
        tif = read_img(dirname + filename)
        tifInfo = tif[:5]
        core = tif[5]
        core = core.reshape(-1, core.shape[2])
        indexes = np.where(core != 15)  # 有效像元筛选条件  用于筛选Buffer的样本区域
        print("Save the indexes")
        np.save(dirname + '\\' + split + '\\construct\\core\\sample_index_' + split + '.npy', indexes[0])
        print("Save the corerClassValue")
        np.save(dirname + '\\' + split + '\\construct\\core\\sample_class.npy', core[indexes[0]])

        filenameList = ["\\" + split + "\\" + year + "_Restoration_15.tif",
                        "\\core\\Slope.tif",
                        "\\core\\Aspect.tif",
                        "\\" + split + "\\" + Builtname + ".tif"]  # 被处理栅格
        for filename in filenameList:
            tif = read_img(dirname + filename)
            tifInfo = tif[:5]
            img = tif[5]
            img = img.reshape(-1, img.shape[2])
            effectivePiex = img[indexes[0]]
            print("Save the effectivePiex")
            savename = filename.split("\\")[len(filename.split("\\")) - 1].split(".")[0]
            np.save(dirname + '\\' + split + '\\construct\\core\\' + savename + '_sample.npy', effectivePiex)


# 用于生成core区域的npy（用于最后的预测）
def core_dataset():
    print("-----------core_dataset-----------------")
    dirname = "D:\\BingqianWang\\RuoergaiClassifity\\data\\"
    filename = year + "\\experiment\\buffer\\buffer_class.tif"  # 有效像元栅格
    tif = read_img(dirname + filename)
    tifInfo = tif[:5]
    buffer = tif[5]
    buffer = buffer.reshape(-1, buffer.shape[2])
    indexes = np.where(buffer == 1)  # 有效像元筛选条件  用于整个Buffer删选
    print("Save the indexes")
    np.save(dirname + year + '\\experiment\\core\\index_core_' + year + '.npy', indexes[0])
    indexes_mount = np.where(buffer == 0)  # 有效像元筛选条件  用于整个Buffer删选
    print("Save the MounteIndex")
    np.save(dirname + year + '\\experiment\\core\\index_mount_' + year + '.npy', indexes_mount[0])
    filenameList = [year + "\\" + year + "_Restoration_15.tif", "core\\Aspect.tif", "core\\Slope.tif",
                    year + "\\" + Builtname + ".tif"]
    for filename in filenameList:
        tif = read_img(dirname + filename)
        tifInfo = tif[:5]
        img = tif[5]
        img = img.reshape(-1, img.shape[2])
        effectivePiex = img[indexes[0]]
        print("Save the effectivePiex")
        savename = filename.split("\\")[len(filename.split("\\")) - 1].split(".")[0]
        np.save(dirname + year + '\\experiment\\core\\' + savename + '.npy', effectivePiex)



if __name__ == "__main__":
    train_visual()
    core_train()
    core_dataset()
