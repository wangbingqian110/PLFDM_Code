from modules import *
import hydra
import torch.multiprocessing
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.data import create_pascal_label_colormap, find_nearest_divisible, sequence2img, Ruoergai_Buffer
from train_segmentation_ruoergai import LitUnsupervisedSegmenter
from tqdm import tqdm

import random

torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(config_path="configs", config_name="demo_config_ruoergaiBuffer.yml")
def my_app(cfg: DictConfig) -> None:
    model = LitUnsupervisedSegmenter(2, cfg)
    model.load_state_dict(torch.load(cfg.model_path))
    upsample = nn.Upsample(size=192, mode='nearest')
    print(OmegaConf.to_yaml(model.cfg))

    dataset = Ruoergai_Buffer(
        cfg=cfg,
        root=cfg.image_dir,
        year=cfg.year,
        image_set=cfg.dataset_name
    )

    Count = dataset.CumputeLength
    batch = cfg.batch_size if Count % cfg.batch_size == 0 else find_nearest_divisible(Count, cfg.batch_size)
    loader = DataLoader(dataset, batch,
                        shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=True, collate_fn=flexible_collate)

    model.eval().cuda()

    par_model = model.net
    linear_probe = model.linear_probe

    colormap = create_pascal_label_colormap()
    outputs = []

    bufferIndex = np.load(cfg.image_dir + "\\bufferIndex.npy")
    coreIndex = np.load(cfg.image_dir + "\\coreIndex.npy")
    tif = read_img('D:\\BingqianWang\\RuoergaiClassifity\\data\\buffer\\buffer_raster.tif')
    tifInfo = tif[:5]
    H = tif[1]
    W = tif[0]
    C = 1

    for img in tqdm(loader, mininterval=10):
        with torch.no_grad():
            img_origin = img[1].cuda()
            img = sequence2img(img_origin.cpu(), cfg.res, cfg.Multi_threaded)
            feat, code = par_model(img, img_origin)
            code = code.reshape(code.shape[0], -1)
            res_code = torch.cat((code, upsample(img_origin.unsqueeze(0)).squeeze(0)), 1)
            linear_preds = linear_probe(res_code)
            linear_preds = linear_preds.argmax(1)
            outputs.append(linear_preds)

    # 预测图
    tensor_out = torch.stack(outputs)
    originShape = [H * W, C]
    tensor = torch.full(originShape, 9)
    tensor_out = tensor_out.reshape(tensor_out.shape[0] * tensor_out.shape[1], C).cpu()
    tensor[bufferIndex] = tensor_out
    tensor[coreIndex] = torch.full([len(coreIndex), C], 1)
    pred = torch.tensor(tensor.reshape(1, H, W, 1), dtype=torch.uint8)
    pred = pred.squeeze(3)
    jpg = colormap[pred][0]
    plt.imshow(jpg)
    # 原图
    # img = torch.stack(origin_img)
    # img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    # img = img.reshape(int(img.shape[0] / 250000), 500, 500, -1)[:, :, :, :3]
    # img = cuttingImg(img, img_width).cpu()
    # plt.imshow(prep_for_plot(img[0].permute(2, 0, 1)))

    plt.show()
    dirname = "D:\\BingqianWang\\RuoergaiClassifity\\data"
    saveTiff(pred.numpy(), tifInfo[0], tifInfo[1], tifInfo[2], tifInfo[3],
             tifInfo[4], dirname + '\\' + cfg.year + '\\experiment\\buffer\\buffer_class.tif')


if __name__ == "__main__":
    prep_args()
    my_app()
