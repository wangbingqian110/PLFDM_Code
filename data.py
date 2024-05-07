import os
import random
import threading
from os.path import join
from queue import Queue

import numpy as np
import torch.multiprocessing
from PIL import Image
from pyts.image import MarkovTransitionField, RecurrencePlot, GramianAngularField
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets.cityscapes import Cityscapes
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import torch.nn.functional as F

from src.utils import percent_linear, toNormal, transTiff2MatIMG, transTiff2Label


def bit_get(val, idx):
    """Gets the bit value.
    Args:
      val: Input value, int or numpy int array.
      idx: Which bit of the input val.
    Returns:
      The "idx"-th bit of input val.
    """
    return (val >> idx) & 1


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


def create_cityscapes_colormap():
    colors = [(128, 64, 128),
              (244, 35, 232),
              (250, 170, 160),
              (230, 150, 140),
              (70, 70, 70),
              (102, 102, 156),
              (190, 153, 153),
              (180, 165, 180),
              (150, 100, 100),
              (150, 120, 90),
              (153, 153, 153),
              (153, 153, 153),
              (250, 170, 30),
              (220, 220, 0),
              (107, 142, 35),
              (152, 251, 152),
              (70, 130, 180),
              (220, 20, 60),
              (255, 0, 0),
              (0, 0, 142),
              (0, 0, 70),
              (0, 60, 100),
              (0, 0, 90),
              (0, 0, 110),
              (0, 80, 100),
              (0, 0, 230),
              (119, 11, 32),
              (0, 0, 0)]
    return np.array(colors)


class Ruoergai(Dataset):
    def __init__(self, cfg, root, year, image_set):
        super(Ruoergai, self).__init__()
        self.split = image_set
        self.root = root
        self.year = year
        self.normalize = torch.nn.functional.normalize
        self.sample = nn.Upsample(size=cfg.res, mode='bilinear', align_corners=False)
        self.builtName = cfg.Builtname

        self.pixValue_img = []
        self.pixValue_label = []

        if self.split == 'train':
            self.effectiveIndex = np.load(join(self.root, "elementTrainIndex.npy"))
            img_origin = np.load(join(self.root, self.year + "_Restoration_15_sample.npy"))[self.effectiveIndex]
            slope = np.load(join(self.root, "Slope_sample.npy"))[self.effectiveIndex]
            aspect = np.load(join(self.root, "Aspect_sample.npy"))[self.effectiveIndex]
            built = np.load(join(self.root, self.builtName + "_sample.npy"))[self.effectiveIndex]
            label = np.load(join(self.root, "sample_class.npy"))[self.effectiveIndex]
        if self.split == 'val':
            self.effectiveIndex = np.load(join(self.root, "elementTextIndex.npy"))
            img_origin = np.load(join(self.root, self.year + "_Restoration_15_sample.npy"))[self.effectiveIndex]
            slope = np.load(join(self.root, "Slope_sample.npy"))[self.effectiveIndex]
            aspect = np.load(join(self.root, "Aspect_sample.npy"))[self.effectiveIndex]
            built = np.load(join(self.root, self.builtName + "_sample.npy"))[self.effectiveIndex]
            label = np.load(join(self.root, "sample_class.npy"))[self.effectiveIndex]
        # T-SNE可视化
        if self.split == 'tsne_Valsual':
            self.effectiveIndex = np.load(join(self.root, "elementTextIndex_tsne.npy"))
            img_origin = np.load(join(self.root, self.year + "_Restoration_15_sample.npy"))[self.effectiveIndex]
            slope = np.load(join(self.root, "Slope_sample.npy"))[self.effectiveIndex]
            aspect = np.load(join(self.root, "Aspect_sample.npy"))[self.effectiveIndex]
            built = np.load(join(self.root, self.builtName + "_sample.npy"))[self.effectiveIndex]
            label = np.load(join(self.root, "sample_class.npy"))[self.effectiveIndex]
        # 用以小方格可视化
        if self.split == 'ruoergai_core':
            slipts = ["clip1", "clip2", 'clip3']
            label = np.array([])
            firstIteration = True
            for split in slipts:
                if firstIteration:
                    img_origin = np.load(
                        join(self.root, "visual_img\\" + self.year + "_Restoration_15_{0}.npy".format(split)))
                    slope = np.load(join("..\\data\\core\\training\\visual_img\\Slope_{0}.npy".format(split)))
                    aspect = np.load(join("..\\data\\core\\training\\visual_img\\Aspect_{0}.npy".format(split)))
                    built = np.load(join(self.root, "visual_img\\" + self.builtName + "_{0}.npy".format(split)))
                    coordinate = np.load(join("..\\data\\core\\training\\visual_img\\coordinate_{0}.npy".format(split)))
                    firstIteration = False
                else:
                    img_origin = np.concatenate(
                        (img_origin, np.load(
                            join(self.root, "visual_img\\" + self.year + "_Restoration_15_{0}.npy".format(split)))),
                        axis=0)
                    slope = np.concatenate(
                        (slope, np.load(join("..\\data\\core\\training\\visual_img\\Slope_{0}.npy".format(split)))),
                        axis=0)
                    aspect = np.concatenate(
                        (aspect, np.load(join("..\\data\\core\\training\\visual_img\\Aspect_{0}.npy".format(split)))),
                        axis=0)
                    built = np.concatenate(
                        (built, np.load(join(self.root, "visual_img\\" + self.builtName + "_{0}.npy".format(split)))),
                        axis=0)
                    coordinate = np.concatenate(
                        (coordinate,
                         np.load(join("..\\data\\core\\training\\visual_img\\coordinate_{0}.npy".format(split)))),
                        axis=0)
        # 用以最后分类
        if self.split == 'demo_core':
            img_origin = np.load(join("../data/" + self.year + "/experiment/core/" + self.year + "_Restoration_15.npy"))
            slope = np.load(join("../data/" + self.year + "/experiment/core/Slope.npy"))
            aspect = np.load(join("../data/" + self.year + "/experiment/core/Aspect.npy"))
            built = np.load(join("../data/" + self.year + "/experiment/core/" + self.builtName + ".npy"))
            label = np.array([])

        # 数据处理
        img = torch.from_numpy(img_origin.astype('int32')).float()
        img = toNormal(img, 0, 65535)
        # if (img == 0).any():
        #     raise
        if cfg.useNDVI:
            img_T = img.permute(1, 0)
            ndvi = (img_T[3] - img_T[2]) / (img_T[3] + img_T[2])
            ndvi = toNormal(ndvi.unsqueeze(1), -1, 1)
            img = torch.cat((img, ndvi), 1)
        if cfg.useNDWI:
            img_T = img.permute(1, 0)
            ndwi = (img_T[1] - img_T[3]) / (img_T[1] + img_T[3])
            ndwi = toNormal(ndwi.unsqueeze(1), -1, 1)
            img = torch.cat((img, ndwi), 1)
        if cfg.useNDMI:
            img_T = img.permute(1, 0)
            ndmi = (img_T[3] - img_T[4]) / (img_T[3] + img_T[4])
            ndmi = toNormal(ndmi.unsqueeze(1), -1, 1)
            img = torch.cat((img, ndmi), 1)
        if cfg.useVARI:
            img_T = img.permute(1, 0)
            vari = (img_T[1] - img_T[2]) / (img_T[1] + img_T[2] - img_T[0])
            # 处理为nan的情况
            nan_mask = torch.isnan(vari)
            vari[nan_mask] = 0
            vari = toNormal(vari.unsqueeze(1), -1, 1)
            img = torch.cat((img, vari), 1)
        if cfg.useSlope:
            slope = torch.from_numpy(slope)
            slope = toNormal(slope, 0, 90)
            img = torch.cat((img, slope), 1)
        if cfg.useAspect:
            aspect = torch.from_numpy(aspect)
            aspect = toNormal(aspect, 0, 360)
            img = torch.cat((img, aspect), 1)
        if cfg.useBuilt:
            built = torch.from_numpy(built.astype('int32'))
            built = toNormal(built, 0, 5769)
            img = torch.cat((img, built), 1)
        if cfg.useCoordinate and self.split == 'ruoergai_core':
            coordinate = torch.from_numpy(coordinate)
            img = torch.cat((img, coordinate), 1)
            coordinate_T = coordinate.permute(1, 0)
            # coordinate_x = coordinate_T[0] / 429780
            # coordinate_y = coordinate_T[1] / 3857190
            # img = torch.cat((img, coordinate_x.unsqueeze(1)), 1)
            # img = torch.cat((img, coordinate_y.unsqueeze(1)), 1)

        self.pixValue_img.append(img)
        self.pixValue_img = torch.cat(self.pixValue_img, 0)
        # self.pixValue_img_normal = torch.nn.functional.normalize(self.pixValue_img, dim=1, p=2)
        self.pixValue_label.append(label)
        self.pixValue_label = np.concatenate(self.pixValue_label, axis=0).astype('int32')

        self.CumputeLength = len(self.pixValue_img)

    def __getitem__(self, index):
        if len(self.pixValue_img[index].shape) == 1:
            img_origin = self.pixValue_img[index].unsqueeze(0)
        else:
            img_origin = self.pixValue_img[index]

        if (True in np.isnan(img_origin)):
            print(img_origin)

        if (len(self.pixValue_label) == 0):  # 在可视化的时候没有Label
            label = np.array([])
        else:
            label = torch.from_numpy(self.pixValue_label[index]).float()

        return label, img_origin.squeeze(0)

    def __len__(self):
        return len(self.pixValue_img)


class Ruoergai_Buffer(Dataset):
    def __init__(self, cfg, root, year, image_set):
        super(Ruoergai_Buffer, self).__init__()
        self.split = image_set
        self.root = root
        self.year = year
        # self.transform = transform
        self.normalize = torch.nn.functional.normalize
        self.sample = nn.Upsample(size=cfg.res, mode='bilinear', align_corners=False)

        self.pixValue_img = []
        self.pixValue_label = []

        if self.split == 'train':
            self.effectiveIndex = np.load(join(self.root, "elementTrainIndex.npy"))
            img_origin = np.load(join(self.root, self.year + "_Restoration_15_sample.npy"))[self.effectiveIndex]
            diffInner = np.load(join(self.root, "EucDist_buff_inner_32bit_sample.npy"))[self.effectiveIndex]
            diffOuter = np.load(join(self.root, "EucDist_buff_outer_32bit_sample.npy"))[self.effectiveIndex]
            relief = np.load(join(self.root, "Relief_500m_sample.npy"))[self.effectiveIndex]
            label = np.load(join(self.root, "buffer_sample_class.npy"))[self.effectiveIndex]
        if self.split == 'val':
            self.effectiveIndex = np.load(join(self.root, "elementTextIndex.npy"))
            img_origin = np.load(join(self.root, self.year + "_Restoration_15_sample.npy"))[self.effectiveIndex]
            diffInner = np.load(join(self.root, "EucDist_buff_inner_32bit_sample.npy"))[self.effectiveIndex]
            diffOuter = np.load(join(self.root, "EucDist_buff_outer_32bit_sample.npy"))[self.effectiveIndex]
            relief = np.load(join(self.root, "Relief_500m_sample.npy"))[self.effectiveIndex]
            label = np.load(join(self.root, "buffer_sample_class.npy"))[self.effectiveIndex]
            # T-SNE可视化
        if self.split == 'tsne_Valsual':
            self.effectiveIndex = np.load(join(self.root, "elementTextIndex_tsne.npy"))
            img_origin = np.load(join(self.root, self.year + "_Restoration_15_sample.npy"))[self.effectiveIndex]
            diffInner = np.load(join(self.root, "EucDist_buff_inner_32bit_sample.npy"))[self.effectiveIndex]
            diffOuter = np.load(join(self.root, "EucDist_buff_outer_32bit_sample.npy"))[self.effectiveIndex]
            relief = np.load(join(self.root, "Relief_500m_sample.npy"))[self.effectiveIndex]
            label = np.load(join(self.root, "buffer_sample_class.npy"))[self.effectiveIndex]
        # 用以小方格可视化
        if self.split == 'ruoergai_buffer':
            slipts = ["clip1", "clip2"]
            img_origin = np.array([])
            diffInner = np.array([])
            diffOuter = np.array([])
            relief = np.array([])
            label = np.array([])
            firstIteration = True
            for split in slipts:
                if firstIteration:
                    img_origin = np.load(
                        join(self.root, "visual_img\\" + self.year + "_Restoration_15_{0}.npy".format(split)))
                    diffInner = np.load(
                        join("../data/buffer/training/visual_img/EucDist_buff_inner_32bit_{0}.npy".format(split)))
                    diffOuter = np.load(
                        join("../data/buffer/training/visual_img/EucDist_buff_outer_32bit_{0}.npy".format(split)))
                    relief = np.load(join("../data/buffer/training/visual_img/Relief_500m_{0}.npy".format(split)))
                    coordinate = np.load(join("../data/buffer/training/visual_img/coordinate_{0}.npy".format(split)))
                    firstIteration = False
                else:
                    img_origin = np.concatenate(
                        (img_origin, np.load(
                            join(self.root, "visual_img\\" + self.year + "_Restoration_15_{0}.npy".format(split)))),
                        axis=0)
                    diffInner = np.concatenate(
                        (diffInner,
                         np.load(join(
                             "../data/buffer/training/visual_img/EucDist_buff_inner_32bit_{0}.npy".format(split)))),
                        axis=0)

                    diffOuter = np.concatenate(
                        (diffOuter,
                         np.load(join(
                             "../data/buffer/training/visual_img/EucDist_buff_outer_32bit_{0}.npy".format(split)))),
                        axis=0)
                    relief = np.concatenate(
                        (relief, np.load(join("../data/buffer/training/visual_img/Relief_500m_{0}.npy".format(split)))),
                        axis=0)
                    coordinate = np.concatenate(
                        (coordinate,
                         np.load(join("../data/buffer/training/visual_img/coordinate_{0}.npy".format(split)))),
                        axis=0)
        # 用以最后分类
        if self.split == 'demo_buffer':
            img_origin = np.load(
                join("../data/" + self.year + "/experiment/buffer/" + self.year + "_Restoration_15.npy"))
            diffInner = np.load(join(self.root, "EucDist_buff_inner_32bit.npy"))
            diffOuter = np.load(join(self.root, "EucDist_buff_outer_32bit.npy"))
            relief = np.load(join(self.root, "Relief_500m.npy"))
            label = np.array([])

        # 数据处理
        img = torch.from_numpy(img_origin.astype('int32')).float()
        img = toNormal(img, 0, 65535)
        if cfg.useDistance:
            diffInner = torch.from_numpy(diffInner)
            diffInner = toNormal(diffInner, 0, 20504)
            img = torch.cat((img, diffInner), 1)
            diffOuter = torch.from_numpy(diffOuter)
            diffOuter = toNormal(diffOuter, 0, 19519)
            img = torch.cat((img, diffOuter), 1)
        if cfg.userelief:
            relief = torch.from_numpy(relief.astype('int32'))
            relief = toNormal(relief, 0, 600)
            img = torch.cat((img, relief), 1)
        if cfg.useCoordinate and self.split == 'ruoergai_buffer':
            coordinate = torch.from_numpy(coordinate)
            img = torch.cat((img, coordinate), 1)

        self.pixValue_img.append(img)
        self.pixValue_img = torch.cat(self.pixValue_img, 0)
        # self.pixValue_img = torch.nn.functional.normalize(self.pixValue_img, dim=1, p=2)
        self.pixValue_label.append(label)
        self.pixValue_label = np.concatenate(self.pixValue_label, axis=0).astype('int32')

        self.CumputeLength = len(self.pixValue_img)

    def __getitem__(self, index):
        if len(self.pixValue_img[index].shape) == 1:
            img_origin = self.pixValue_img[index].unsqueeze(0)
        else:
            img_origin = self.pixValue_img[index]

        if (True in np.isnan(img_origin)):
            print(img_origin)

        if (len(self.pixValue_label) == 0):  # 在可视化的时候没有Label
            label = np.array([])
        else:
            label = torch.from_numpy(self.pixValue_label[index]).float()

        return label, (img_origin).squeeze(0)

    def __len__(self):
        return len(self.pixValue_img)


class ContrastiveSegDataset(Dataset):
    def __init__(self,
                 pytorch_data_dir,
                 dataset_name,
                 crop_type,
                 image_set,
                 transform,
                 target_transform,
                 cfg,
                 aug_geometric_transform=None,
                 aug_photometric_transform=None,
                 num_neighbors=5,
                 compute_knns=False,
                 mask=False,
                 pos_labels=False,
                 pos_images=False,
                 extra_transform=None,
                 model_type_override=None,
                 useNNS=True
                 ):
        super(ContrastiveSegDataset).__init__()
        self.num_neighbors = num_neighbors
        self.image_set = image_set
        self.dataset_name = dataset_name
        self.mask = mask
        self.pos_labels = pos_labels
        self.pos_images = pos_images
        self.extra_transform = extra_transform

        if dataset_name == "ruoergai_core":
            self.n_classes = 7
            dataset_class = Ruoergai
            extra_args = dict(coarse_labels=True)
        elif dataset_name == "ruoergai_buffer":
            self.n_classes = 2
            dataset_class = Ruoergai_Buffer
            extra_args = dict(coarse_labels=True)
        else:
            raise ValueError("Unknown dataset: {}".format(dataset_name))

        self.aug_geometric_transform = aug_geometric_transform
        self.aug_photometric_transform = aug_photometric_transform

        self.dataset = dataset_class(
            cfg=cfg,
            root=pytorch_data_dir,
            year=cfg.year,
            image_set=self.image_set
        )

        if model_type_override is not None:
            model_type = model_type_override
        else:
            model_type = cfg.model_type

        if useNNS == True:
            # nice_dataset_name = cfg.dir_dataset_name if dataset_name == "directory" else dataset_name
            nearest_feature_cache_file = join(pytorch_data_dir, "nns/nearest_{}".format(cfg.nns_file_name_inter))
            intra_feature_cache_file = join(pytorch_data_dir, "nns/nearest_{}".format(cfg.nns_file_name_intra))
            farest_feature_cache_file = join(pytorch_data_dir, "nns/farthest_{}".format(cfg.nns_file_name_neg))
            if pos_labels or pos_images:
                if not os.path.exists(nearest_feature_cache_file) or compute_knns:
                    raise ValueError(
                        "could not find nn file {} please run precompute_knns".format(nearest_feature_cache_file))
                else:
                    loaded_nearest = np.load(nearest_feature_cache_file)
                    loaded_intra = np.load(intra_feature_cache_file)
                    loaded_neg = np.load(farest_feature_cache_file)
                    self.nns_nearest = loaded_nearest["nns"]
                    self.nns_intra = loaded_intra["nns"]
                    self.nns_neg = loaded_neg["nns"]
                assert len(self.dataset) == self.nns_nearest.shape[0]

    def __len__(self):
        return len(self.dataset)

    def _set_seed(self, seed):
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7

    def __getitem__(self, ind):
        pack = self.dataset[ind]

        if self.pos_images or self.pos_labels:
            ind_pos = self.nns_nearest[ind]
            pack_pos = self.dataset[ind_pos]
            index_intra = self.nns_intra[ind][torch.randint(low=1, high=self.num_neighbors + 1, size=[]).item()]
            pack_intra = self.dataset[index_intra]
            index_neg = self.nns_neg[ind]
            pack_neg = self.dataset[index_neg]

        if self.extra_transform is not None:
            extra_trans = self.extra_transform
        else:
            extra_trans = lambda i, x: x

        ret = {
            "img_origin": extra_trans(ind, pack[1]),
            "label": extra_trans(ind, pack[0]),
        }

        if self.pos_images:
            ret["orgin_pos"] = extra_trans(ind, pack_pos[1])
            ret["intra_pos"] = extra_trans(ind, pack_intra[1])
            ret["orgin_neg"] = extra_trans(ind, pack_neg[1])

        if self.pos_labels:
            ret["label_pos"] = extra_trans(ind, pack_pos[0])

        return ret


def find_nearest_divisible(number, divisor):
    closest_divisible = divisor

    # 逐步增加或减少给定数，直到找到能够整除定值的数
    while number % closest_divisible != 0:
        closest_divisible -= 1  # 或者使用 closest_divisible -= 1

    return closest_divisible


def sequence2img(img_origin, res, Multi_threaded):
    sample = nn.Upsample(size=res, mode='bilinear', align_corners=False)

    def gasf(img_origin, imgList):
        gasf = GramianAngularField(method='summation')
        img_1 = torch.from_numpy(gasf.transform(img_origin.numpy()))
        img_1 = sample(img_1.unsqueeze(1))
        imgList.append(img_1)

    def mtf(img_origin, imgList):
        mtf = MarkovTransitionField(n_bins=2)
        img_2 = torch.from_numpy(mtf.transform(img_origin.numpy()))
        img_2 = sample(img_2.unsqueeze(1))
        imgList.append(img_2)

    def rp(img_origin, imgList):
        rp = RecurrencePlot(dimension=1, time_delay=1)
        img_3 = torch.from_numpy(rp.transform(img_origin.numpy()))
        img_3 = sample(img_3.unsqueeze(1))
        imgList.append(img_3)

    imgList = []
    # 是否多线程处理
    if Multi_threaded == True:
        # 线程列表
        threads = []
        t_gasf = threading.Thread(target=gasf, args=(img_origin, imgList))
        t_gasf.start()
        threads.append(t_gasf)
        t_mtf = threading.Thread(target=mtf, args=(img_origin, imgList))
        t_mtf.start()
        threads.append(t_mtf)
        t_rp = threading.Thread(target=rp, args=(img_origin, imgList))
        t_rp.start()
        threads.append(t_rp)

        # # 对所有线程进行阻塞
        for thread in threads:
            thread.join()
    else:
        gasf(img_origin, imgList)
        mtf(img_origin, imgList)
        rp(img_origin, imgList)

    img = torch.cat((imgList[0], imgList[1], imgList[2]), 1).float()
    return img.cuda()
