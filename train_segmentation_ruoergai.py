from torchmetrics import ROC, Accuracy, CohenKappa
from torchmetrics.classification import MulticlassConfusionMatrix, BinaryPrecisionRecallCurve, BinaryROC, \
    MulticlassF1Score

from modules import *
from data import *
from torch.utils.data import DataLoader
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torch.multiprocessing
import sys
import warnings

warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy('file_system')


# 保存实验结果为tiff
def save2tiff(cfg, dirName, pred, step, index):
    dirPath = "../logs/{}/{}".format(cfg.full_name, dirName)
    os.makedirs(dirPath, exist_ok=True)
    prefix = dirPath + "/{}_clip{}.tif".format(step, index)
    tif_ref = read_img(
        "D:\\BingqianWang\\RuoergaiClassifity\\data\\core\\training\\visual_img\\Slope_clip" + str(index + 1) + ".tif")
    tifInfo = tif_ref[:5]
    saveTiff(pred.numpy(), tifInfo[0], tifInfo[1], tifInfo[2], tifInfo[3],
             tifInfo[4], prefix)


# 保存实验结果至文件
def save2file(cfg, dirName, OA, Kappa, F1_Class, step):
    dirPath = "../logs/{}/{}".format(cfg.full_name, dirName)
    os.makedirs(dirPath, exist_ok=True)
    # OA_liner = tb_metrics['test/liner_Accuracy'].item()
    # OA_cluster = tb_metrics['test/cluster_Accuracy']
    F1_Class = [" ".join(row) for row in F1_Class.cpu().numpy().astype(str)]
    # 要写入文件的内容
    text_to_save = """-----------step: {}---------------
    OA: {},
    Kappa: {},
    F1: {}
    
    """.format(step, OA, Kappa, F1_Class)
    # 打开文件进行写入，如果文件不存在将会被创建
    prefix = dirPath + "/result.txt"
    with open(prefix, 'a', encoding='utf-8') as file:
        file.write(text_to_save)  # 将文本写入文件


# 绘制混淆矩阵的函数
def plot_confusion_matrix(cm, labels_name, is_norm=True, colorbar=True, cmap=plt.cm.Blues):
    plt.figure(dpi=100)
    if is_norm == True:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)  # 横轴归一化并保留2位小数

    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # 在特定的窗口上显示图像
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')  # 默认所有值均为黑色
            # plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', color="white" if i==j else "black", verticalalignment='center') # 将对角线值设为白色
    if colorbar:
        plt.colorbar()  # 创建颜色条

    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=45)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    # plt.title(title)  # 图像标题
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def getClipResult(tensor, index_clip1, index_clip2, index_clip3):
    # 根据索引文件将整个的输出结果分到各自的方格中
    tensor = tensor.reshape(tensor.shape[0] * tensor.shape[1], 1).cpu()
    img_clip1 = tensor[0:index_clip1.size]
    img_clip2 = tensor[index_clip1.size:index_clip1.size + index_clip2.size]
    after_clip1 = getClipImg(index_clip1, img_clip1).permute(1, 2, 0).squeeze(2)
    after_clip2 = getClipImg(index_clip2, img_clip2).permute(1, 2, 0).squeeze(2)
    if len(index_clip3) > 0:
        img_clip3 = tensor[index_clip1.size + index_clip2.size:index_clip1.size + index_clip2.size + index_clip3.size]
        after_clip3 = getClipImg(index_clip3, img_clip3).permute(1, 2, 0).squeeze(2)
    else:
        after_clip3 = []
    return after_clip1, after_clip2, after_clip3


def get_class_labels(dataset_name):
    if dataset_name.startswith("cityscapes"):
        return [
            'road', 'sidewalk', 'parking', 'rail track', 'building',
            'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'caravan', 'trailer', 'train',
            'motorcycle', 'bicycle']
    elif dataset_name == "cocostuff27":
        return [
            "electronic", "appliance", "food", "furniture", "indoor",
            "kitchen", "accessory", "animal", "outdoor", "person",
            "sports", "vehicle", "ceiling", "floor", "food",
            "furniture", "rawmaterial", "textile", "wall", "window",
            "building", "ground", "plant", "sky", "solid",
            "structural", "water"]
    elif dataset_name == "voc":
        return [
            'background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif dataset_name == "potsdam":
        return [
            'roads and cars',
            'buildings and clutter',
            'trees and vegetation']
    else:
        raise ValueError("Unknown Dataset {}".format(dataset_name))


def cuttingImg(img, patchsize):
    B, H, W, C = img.shape
    H_r, W_r = H % patchsize, W % patchsize
    img = img[:B, :H - H_r, :W - W_r, :C]
    cuted = img.reshape(B, H // patchsize, patchsize, W // patchsize, patchsize, C)
    cuted = cuted.permute(0, 1, 3, 2, 4, 5).reshape(-1, patchsize, patchsize, C)
    return cuted


def getClipImg(index_clip, img_clip):
    originShape = [1024 * 1024, 1]
    tensor = torch.full(originShape, 4, dtype=torch.int64)
    tensor[index_clip] = img_clip
    result = tensor.reshape(1024, 1024, 1).permute(2, 0, 1)
    result = result.to(torch.uint8)
    return result


class LitUnsupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        if not cfg.continuous:
            dim = n_classes
        else:
            dim = cfg.dim

        self.net = DinoFeaturizer(dim, cfg)
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.eval().cuda()
        self.net.load_state_dict(torch.load(cfg.code_model_path))

        self.cluster_probe = ClusterLookup(220, n_classes + cfg.extra_clusters)
        self.linear_probe = nn.Linear(220, n_classes)
        self.linear_probe_feats = nn.Linear(
            28, n_classes)
        self.linear_probe_origin = nn.Linear(9, n_classes)
        self.linear_probe_three3D = nn.Linear(self.cfg.res ** 2 * 3, n_classes)
        # self.cluster_probe = ClusterLookup(int((self.cfg.res / self.cfg.dino_patch_size) ** 2 * dim),
        #                                    n_classes + cfg.extra_clusters)
        # self.linear_probe = torch.nn.Sequential(
        #     torch.nn.Linear(int((self.cfg.res / self.cfg.dino_patch_size) ** 2 * dim), 80),
        #     torch.nn.Linear(80, n_classes),
        #     # torch.nn.LayerNorm(n_classes),
        # )

        # self.linear_probe_feats = torch.nn.Sequential(
        #     torch.nn.Linear(int((self.cfg.res / self.cfg.dino_patch_size) ** 2 * 384), 80),
        #     torch.nn.Linear(80, n_classes),
        #     # torch.nn.LayerNorm(n_classes),
        # )
        #
        # self.linear_probe_origin = torch.nn.Sequential(
        #     torch.nn.Linear(self.cfg.feature_num, 80),
        #     torch.nn.Linear(80, n_classes),
        # )

        self.linear_metrics = Accuracy(task="multiclass", num_classes=n_classes)
        self.linear_cohenkappa = CohenKappa(task="multiclass", num_classes=n_classes)
        self.linear_multiclassF1Score = MulticlassF1Score(num_classes=n_classes, average=None)

        self.linear_feats_metrics = UnsupervisedMetrics(
            "test/linear_feats/", n_classes, 0, False)
        self.feats_cohenkappa = CohenKappa(task="multiclass", num_classes=n_classes)
        self.feats_multiclassF1Score = MulticlassF1Score(num_classes=n_classes, average=None)
        self.confmat_feats_metrics = MulticlassConfusionMatrix(num_classes=n_classes, normalize='true')

        self.three3D_cohenkappa = CohenKappa(task="multiclass", num_classes=n_classes)
        self.three3D_multiclassF1Score = MulticlassF1Score(num_classes=n_classes, average=None)
        self.linear_three3D_metrics = UnsupervisedMetrics(
            "test/linear_three3D/", n_classes, 0, False)

        self.cluster_metrics = UnsupervisedMetrics(
            "test/cluster/", n_classes, cfg.extra_clusters, True)

        self.linear_origin_metrics = UnsupervisedMetrics(
            "test/linear_orgin/", n_classes, 0, False)

        # 损失函数
        self.linear_probe_loss_fn = torch.nn.CrossEntropyLoss()

        self.automatic_optimization = False

        if self.cfg.dataset_name.startswith("cityscapes"):
            self.label_cmap = create_cityscapes_colormap()
        else:
            self.label_cmap = create_pascal_label_colormap()

        self.val_steps = 0
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.output_num = random.sample(range(0, 1 * 4), self.cfg.n_images)
        self.upsample = nn.Upsample(size=192, mode='nearest')
        plt.rcParams['font.sans-serif'] = 'times new roman'  # 设置全局字体，会被局部字体顶替

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)[1]

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        linear_probe_optim, linear_probe_feats_optim, linear_probe_origin_optim, cluster_probe_optim, linear_probe_three3D_optim = self.optimizers()

        linear_probe_optim.zero_grad()
        linear_probe_feats_optim.zero_grad()
        linear_probe_origin_optim.zero_grad()
        cluster_probe_optim.zero_grad()
        linear_probe_three3D_optim.zero_grad()

        with torch.no_grad():
            img_origin = batch["img_origin"]
            label = batch["label"]

        img = sequence2img(img_origin.cpu(), self.cfg.res, self.cfg.Multi_threaded)
        self.net.eval()
        feats, code = self.net(img, img_origin)
        log_args = dict(sync_dist=False, rank_zero_only=True)

        loss = 0

        flat_label = label.reshape(-1).int()
        mask = (flat_label >= 0) & (flat_label < self.n_classes)

        detached_code = torch.clone(code.detach())
        detached_code = detached_code.reshape(detached_code.shape[0], -1)
        res_code = torch.cat((detached_code, self.upsample(img_origin.unsqueeze(0)).squeeze(0)), 1)
        linear_logits = self.linear_probe(res_code)
        flat_label_onegot = one_hot_feats(flat_label[mask].to(torch.int64), self.n_classes)
        linear_loss = self.linear_probe_loss_fn(linear_logits[mask], flat_label_onegot).mean()
        loss += linear_loss
        self.log('loss/linear', linear_loss, **log_args)

        detached_feats = torch.clone(feats.detach())
        detached_feats = detached_feats.reshape(detached_feats.shape[0], -1)
        linear_logits_feats = self.linear_probe_feats(detached_code)
        linear_loss_feats = self.linear_probe_loss_fn(linear_logits_feats[mask], flat_label_onegot).mean()
        loss += linear_loss_feats
        self.log('loss/linear_feats', linear_loss_feats, **log_args)

        detached_orgin = torch.clone(img_origin.detach())
        linear_logits_orgin = self.linear_probe_origin(detached_orgin)
        linear_loss_orgin = self.linear_probe_loss_fn(linear_logits_orgin[mask], flat_label_onegot).mean()
        loss += linear_loss_orgin
        self.log('loss/linear_orgin', linear_loss_orgin, **log_args)

        threeD = torch.clone(img.detach())
        threeD = threeD.reshape(threeD.shape[0], -1)
        linear_logits_three3D = self.linear_probe_three3D(threeD)
        linear_loss_three3D = self.linear_probe_loss_fn(linear_logits_three3D[mask], flat_label_onegot).mean()
        loss += linear_loss_three3D
        self.log('loss/linear_three3D', linear_loss_three3D, **log_args)

        cluster_loss, cluster_probs = self.cluster_probe(res_code, 1)
        loss += cluster_loss
        self.log('loss/cluster', cluster_loss, **log_args)

        self.log('loss/total', loss, **log_args)

        self.manual_backward(loss)
        cluster_probe_optim.step()
        linear_probe_optim.step()
        linear_probe_origin_optim.step()
        linear_probe_feats_optim.step()
        linear_probe_three3D_optim.step()

        return loss

    def on_train_start(self):
        print('Train start')

    def validation_step(self, batch, batch_idx):
        label = batch["label"]
        img_origin = batch["img_origin"]
        img = sequence2img(img_origin.cpu(), self.cfg.res, self.cfg.Multi_threaded)
        self.net.eval()

        with torch.no_grad():
            # code = self.net(img)
            feats, code = self.net(img, img_origin)

            code = code.reshape(code.shape[0], -1)
            res_code = torch.cat((code, self.upsample(img_origin.unsqueeze(0)).squeeze(0)), 1)
            linear_preds = self.linear_probe(res_code)
            linear_preds = linear_preds.argmax(1)
            self.linear_metrics.update(linear_preds, torch.tensor(label, dtype=torch.int8).squeeze(1))
            self.linear_multiclassF1Score.update(linear_preds, torch.tensor(label, dtype=torch.int8).squeeze(1))
            self.linear_cohenkappa.update(linear_preds, torch.tensor(label, dtype=torch.int8).squeeze(1))

            feats = feats.reshape(feats.shape[0], -1)
            linear_preds_feats = self.linear_probe_feats(code)
            linear_preds_feats = linear_preds_feats.argmax(1)
            self.linear_feats_metrics.update(linear_preds_feats, torch.tensor(label, dtype=torch.int8))
            self.feats_multiclassF1Score.update(linear_preds, torch.tensor(label, dtype=torch.int8).squeeze(1))
            self.feats_cohenkappa.update(linear_preds, torch.tensor(label, dtype=torch.int8).squeeze(1))
            self.confmat_feats_metrics.update(linear_preds_feats, torch.tensor(label, dtype=torch.int8).squeeze(1))

            linear_preds_origin = self.linear_probe_origin(img_origin)
            linear_preds_origin = linear_preds_origin.argmax(1)
            self.linear_origin_metrics.update(linear_preds_origin, torch.tensor(label, dtype=torch.int8))

            three3D = img.reshape(img.shape[0], -1)
            linear_preds_three3D = self.linear_probe_three3D(three3D)
            linear_preds_three3D = linear_preds_three3D.argmax(1)
            self.linear_three3D_metrics.update(linear_preds_three3D, torch.tensor(label, dtype=torch.int8))
            self.three3D_multiclassF1Score.update(linear_preds_three3D,
                                                  torch.tensor(label, dtype=torch.int8).squeeze(1))
            self.three3D_cohenkappa.update(linear_preds_three3D, torch.tensor(label, dtype=torch.int8).squeeze(1))

            cluster_loss, cluster_preds = self.cluster_probe(res_code, 1)
            cluster_preds = cluster_preds.argmax(1)
            self.cluster_metrics.update(cluster_preds, torch.tensor(label, dtype=torch.int8).squeeze(1))

            self.validation_step_outputs.append({
                'img': img_origin.detach().cpu(),
                'linear_preds': linear_preds.detach().cpu(),
                'linear_preds_feats': linear_preds_feats.detach().cpu(),
                'linear_preds_origin': linear_preds_origin.detach().cpu(),
                "cluster_preds": cluster_preds.detach().cpu(),
                "label": label.detach().cpu()
            })
            return {
                'img': img_origin.detach().cpu(),
                'linear_preds': linear_preds.detach().cpu(),
                'linear_preds_feats': linear_preds_feats.detach().cpu(),
                'linear_preds_origin': linear_preds_origin.detach().cpu(),
                "cluster_preds": cluster_preds.detach().cpu(),
                "label": label.detach().cpu()}

    def on_validation_epoch_end(self) -> None:
        tb_metrics = {
            'test/liner_Accuracy': self.linear_metrics.compute(),
            'test/cluster_Accuracy': self.cluster_metrics.compute()[0],
            'test/feats_Accuracy': self.linear_feats_metrics.compute()[0],
            # **self.linear_origin_metrics.compute(),
            'test/linear_three3D': self.linear_three3D_metrics.compute()[0],
        }

        if self.trainer.is_global_zero and not self.cfg.submitting_to_aml:
            if self.cfg.dataset_name == 'ruoergai_buffer':
                Visual_dataset = Ruoergai_Buffer(
                    cfg=self.cfg,
                    root=self.cfg.pytorch_data_dir,
                    image_set=self.cfg.dataset_name,
                    year=self.cfg.year,
                )
            elif self.cfg.dataset_name == 'ruoergai_core':
                Visual_dataset = Ruoergai(
                    cfg=self.cfg,
                    root=self.cfg.pytorch_data_dir,
                    image_set=self.cfg.dataset_name,
                    year=self.cfg.year,
                )
            Count = Visual_dataset.CumputeLength
            batch = self.cfg.val_batch_size if Count % self.cfg.val_batch_size == 0 else find_nearest_divisible(
                Count,
                self.cfg.val_batch_size)
            loader = DataLoader(Visual_dataset, batch,
                                shuffle=False, num_workers=self.cfg.num_workers,
                                pin_memory=True, collate_fn=flexible_collate)
            outputs = []
            outputs_orgin = []
            outputs_feats = []
            outputs_cluster = []
            outputs_three3D = []

            for batch in tqdm(loader, dynamic_ncols=True, mininterval=10):
                img_origin = batch[1].cuda()
                img = sequence2img(img_origin.cpu(), self.cfg.res, self.cfg.Multi_threaded)
                # self.load_state_dict(torch.load('../checkpoints/core_train/classifity/core_train_8000_all_params.pth'))
                with torch.no_grad():
                    # 线性探针
                    feat, code = self.net(img, img_origin)
                    code = code.reshape(code.shape[0], -1)
                    res_code = torch.cat((code, self.upsample(img_origin.unsqueeze(0)).squeeze(0)), 1)
                    linear_preds = self.linear_probe(res_code)
                    linear_preds = linear_preds.argmax(1)
                    outputs.append(linear_preds)
                    # 冻结模型探针
                    feats = feat.reshape(feat.shape[0], -1)
                    linear_preds_feats = self.linear_probe_feats(code)
                    linear_preds_feats = linear_preds_feats.argmax(1)
                    outputs_feats.append(linear_preds_feats)
                    # 原始探针
                    linear_preds_origin = self.linear_probe_origin(img_origin)
                    linear_preds_origin = linear_preds_origin.argmax(1)
                    outputs_orgin.append(linear_preds_origin)
                    # 3D图像探针
                    three3D = img.reshape(img.shape[0], -1)
                    linear_preds_three3D = self.linear_probe_three3D(three3D)
                    linear_preds_three3D = linear_preds_three3D.argmax(1)
                    outputs_three3D.append(linear_preds_three3D)
                    # 聚类探针
                    cluster_loss, cluster_preds = self.cluster_probe(res_code, 1)
                    cluster_preds = cluster_preds.argmax(1)
                    outputs_cluster.append(cluster_preds)

            if self.cfg.dataset_name == 'ruoergai_buffer':
                index_clip1 = np.load(join("../data/buffer/training/visual_img/index_clip1.npy"))
                index_clip2 = np.load(join("../data/buffer/training/visual_img/index_clip2.npy"))
                index_clip3 = []
            elif self.cfg.dataset_name == 'ruoergai_core':
                index_clip1 = np.load("..\\data\\core\\training\\visual_img\\index_clip1.npy")
                index_clip2 = np.load("..\\data\\core\\training\\visual_img\\index_clip2.npy")
                index_clip3 = np.load("..\\data\\core\\training\\visual_img\\index_clip3.npy")
            tensor_linear = torch.stack(outputs)
            tensor_orgin = torch.stack(outputs_orgin)
            tensor_cluster = torch.stack(outputs_cluster)
            tensor_feat = torch.stack(outputs_feats)
            tensor_three3D = torch.stack(outputs_three3D)

            linear_clip = getClipResult(tensor_linear, index_clip1, index_clip2, index_clip3)
            orgin_clip = getClipResult(tensor_orgin, index_clip1, index_clip2, index_clip3)
            cluster_clip = getClipResult(tensor_cluster, index_clip1, index_clip2, index_clip3)
            feats_clip = getClipResult(tensor_feat, index_clip1, index_clip2, index_clip3)
            three3D_clip = getClipResult(tensor_three3D, index_clip1, index_clip2, index_clip3)

            fig, ax = plt.subplots(5, self.cfg.n_images, figsize=(self.cfg.n_images * 3, 4 * 3))
            for index in range(self.cfg.n_images):
                ax[0, index].imshow(self.label_cmap[linear_clip[index]])
                ax[1, index].imshow(self.label_cmap[orgin_clip[index]])
                ax[2, index].imshow(self.label_cmap[self.cluster_metrics.map_clusters(
                    cluster_clip[index]) if self.cfg.dataset_name == 'ruoergai_core' else cluster_clip[index]])
                ax[3, index].imshow(self.label_cmap[feats_clip[index]])
                ax[4, index].imshow(self.label_cmap[three3D_clip[index]])

            ax[0, 0].set_ylabel("Visual-linear", fontsize=16)
            ax[1, 0].set_ylabel("Visual-orgin", fontsize=16)
            ax[2, 0].set_ylabel("Visual-cluster", fontsize=16)
            ax[3, 0].set_ylabel("Visual-feats", fontsize=16)
            ax[4, 0].set_ylabel("Visual-three3D", fontsize=16)
            remove_axes(ax)
            plt.tight_layout()
            add_plot(self.logger.experiment, "plot_labels", int(self.global_step / 5))
            # 绘制混合矩阵图
            cmtx = self.confmat_feats_metrics.compute(),
            # figure = plt.figure()
            plot_confusion_matrix(cmtx[0].cpu().numpy(),
                                  ["River-lake", "Marsh", "Swamp meadow", "Meadow", "Meadow-shrub-arbor", "Bare land",
                                   "Building-road"])
            add_plot(self.logger.experiment, "ConfusionMatrix", int(self.global_step / 5))
            # 实验结果写入文件
            F1_linear = self.linear_multiclassF1Score.compute()
            save2file(self.cfg, "linear", tb_metrics['test/liner_Accuracy'], self.linear_cohenkappa.compute(),
                      F1_linear, int(self.global_step / 5))
            F1_three3D = self.three3D_multiclassF1Score.compute()
            save2file(self.cfg, "three3D", tb_metrics['test/linear_three3D'], self.three3D_cohenkappa.compute(),
                      F1_three3D, int(self.global_step / 5))
            F1_cluster = self.cluster_metrics.compute()[1]
            save2file(self.cfg, "cluster", tb_metrics['test/cluster_Accuracy'], self.cluster_metrics.compute()[2],
                      F1_cluster, int(self.global_step / 5))
            F1_feats = self.feats_multiclassF1Score.compute()
            save2file(self.cfg, "feats", tb_metrics['test/feats_Accuracy'], self.feats_cohenkappa.compute(),
                      F1_feats, int(self.global_step / 5))
            # 保存实验结果tif
            for index in range(self.cfg.n_images):
                save2tiff(self.cfg, "linear", linear_clip[index], int(self.global_step / 5), index)
            for index in range(self.cfg.n_images):
                save2tiff(self.cfg, "three3D", three3D_clip[index], int(self.global_step / 5), index)
            for index in range(self.cfg.n_images):
                save2tiff(self.cfg, "feats", feats_clip[index], int(self.global_step / 5), index)
            for index in range(self.cfg.n_images):
                save2tiff(self.cfg, "cluster", self.cluster_metrics.map_clusters(
                    cluster_clip[index]) if self.cfg.dataset_name == 'ruoergai_core' else cluster_clip[index],
                          int(self.global_step / 5), index)

        if int(self.global_step / 5) > 2:
            torch.save(self.state_dict(),
                       join(self.cfg.output_root, "checkpoints/" + self.cfg.experiment_name + "/classifity",
                            '{}_{}_all_params.pth'.format(self.cfg.experiment_name, int(self.global_step / 5))))
            self.log_dict(tb_metrics)

        self.linear_metrics.reset()
        self.linear_multiclassF1Score.reset()
        self.linear_cohenkappa.reset()

        self.linear_feats_metrics.reset()
        self.feats_multiclassF1Score.reset()
        self.feats_cohenkappa.reset()
        self.confmat_feats_metrics.reset()

        self.cluster_metrics.reset()

        self.linear_three3D_metrics.reset()
        self.three3D_multiclassF1Score.reset()
        self.three3D_cohenkappa.reset()

        self.linear_origin_metrics.reset()

    def configure_optimizers(self):
        linear_probe_optim = torch.optim.Adam(list(self.linear_probe.parameters()), lr=0.001)
        linear_probe_feats_optim = torch.optim.Adam(list(self.linear_probe_feats.parameters()), lr=0.005)
        linear_probe_origin_optim = torch.optim.Adam(list(self.linear_probe_origin.parameters()), lr=0.1)
        cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=0.005)
        linear_probe_three3D_optim = torch.optim.Adam(list(self.linear_probe_three3D.parameters()), lr=0.005)

        return linear_probe_optim, linear_probe_feats_optim, linear_probe_origin_optim, cluster_probe_optim, linear_probe_three3D_optim


@hydra.main(config_path="configs", config_name="train_config_ruoergaiCore.yml")
def my_app(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = cfg.pytorch_data_dir
    data_dir = join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    checkpoint_dir = join(cfg.output_root, "checkpoints")

    prefix = "{}/{}_{}".format(cfg.log_dir + '/classifity-train', cfg.dataset_name, cfg.experiment_name)
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
    cfg.full_name = name

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # seed_everything(seed=0)

    print(data_dir)
    print(cfg.output_root)

    geometric_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=cfg.res, scale=(0.8, 1.0))
    ])
    photometric_transforms = T.Compose([
        T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        T.RandomGrayscale(.2),
        T.RandomApply([T.GaussianBlur((5, 5))])
    ])

    sys.stdout.flush()

    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=get_transform(cfg.res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.res, True, cfg.loader_crop_type),
        cfg=cfg,
        aug_geometric_transform=geometric_transforms,
        aug_photometric_transform=photometric_transforms,
        num_neighbors=cfg.num_neighbors,
        useNNS=False
    )

    if cfg.dataset_name == "voc":
        val_loader_crop = None
    else:
        val_loader_crop = "center"

    val_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(224, False, val_loader_crop),
        target_transform=get_transform(224, True, val_loader_crop),
        mask=True,
        cfg=cfg,
        useNNS=False
    )

    # val_dataset = MaterializedDataset(val_dataset)
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    if cfg.submitting_to_aml:
        val_batch_size = 16
    else:
        val_batch_size = cfg.val_batch_size

    val_loader = DataLoader(val_dataset, val_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = LitUnsupervisedSegmenter(train_dataset.n_classes, cfg)

    tb_logger = TensorBoardLogger(
        join(log_dir, name),
        default_hp_metric=False
    )

    trainer = Trainer(
        accelerator='gpu',
        num_sanity_val_steps=1,
        log_every_n_steps=cfg.scalar_log_freq,
        logger=tb_logger,
        max_steps=cfg.max_steps,
        check_val_every_n_epoch=None,
        val_check_interval=cfg.val_freq
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    prep_args()
    my_app()
