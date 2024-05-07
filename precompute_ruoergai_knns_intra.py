from sklearn.manifold import TSNE

from data import ContrastiveSegDataset, Ruoergai_Buffer
from modules import *
import os
from os.path import join
import hydra
import numpy as np
import torch.multiprocessing
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data import sequence2img, Ruoergai


def get_feats(model, loader, cfg):
    all_pix = []
    for pack in tqdm(loader):
        img_origin = pack[1]
        img = sequence2img(img_origin.cpu(), cfg.res, cfg.Multi_threaded)
        feats, code = model(img.cuda(), img_origin.cuda())
        feats = TSNE(n_components=2, random_state=0).fit_transform(feats.cpu())
        # feats = F.avg_pool1d(feats, kernel_size=cfg.kernel_size_feat_pool, stride=cfg.kernel_size_feat_pool)
        feats = F.normalize(torch.from_numpy(feats))
        all_pix.append(feats.to("cuda:0", non_blocking=True))
    return torch.cat(all_pix, dim=0).contiguous()


@hydra.main(config_path="configs", config_name="train_config_ruoergaiCore.yml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = cfg.pytorch_data_dir
    data_dir = join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(join(pytorch_data_dir, "nns"), exist_ok=True)

    print(data_dir)
    print(cfg.output_root)

    image_sets = [cfg.dataset_name]
    dataset_names = [cfg.dataset_name]
    crop_types = [None]

    # Uncomment these lines to run on custom datasets
    # dataset_names = ["directory"]
    # crop_types = [None]

    nns_file_name_Near = cfg.nns_file_name_intra
    nns_file_name_Far = cfg.nns_file_name_neg
    n_batches = 80000

    # if cfg.arch == "dino":
    #     from modules import DinoFeaturizer, LambdaLayer
    #     no_ap_model = torch.nn.Sequential(
    #         DinoFeaturizer(20, cfg),  # dim doesent matter
    #         LambdaLayer(lambda p: p[0]),
    #     ).cuda()
    # else:
    #     cut_model = load_model(cfg.model_type, join(cfg.output_root, "data")).cuda()
    #     no_ap_model = nn.Sequential(*list(cut_model.children())[:-1]).cuda()
    # par_model = torch.nn.DataParallel(no_ap_model)
    par_model = torch.nn.DataParallel(DinoFeaturizer(20, cfg))
    par_model.eval()
    for image_set in image_sets:
        for dataset_name in dataset_names:
            nice_dataset_name = cfg.dir_dataset_name if dataset_name == "directory" else dataset_name

            nearest_feature_cache_file = join(pytorch_data_dir, "nns", "nearest_{}".format(
                nns_file_name_Near))
            farthest_feature_cache_file = join(pytorch_data_dir, "nns", "farthest_{}".format(
                nns_file_name_Far))

            if not os.path.exists(nearest_feature_cache_file):
                print("{} not found, computing".format(nearest_feature_cache_file))
                if image_set == 'ruoergai_buffer':
                    dataset = Ruoergai_Buffer(
                        root=pytorch_data_dir,
                        image_set=image_set,
                        year=cfg.year,
                        cfg=cfg,
                    )
                elif image_set == 'ruoergai_core':
                    dataset = Ruoergai(
                        root=pytorch_data_dir,
                        image_set=image_set,
                        year=cfg.year,
                        cfg=cfg,
                    )

                # loader = DataLoader(dataset, 64, shuffle=False, num_workers=cfg.num_workers, pin_memory=False)

                with torch.no_grad():
                    normed_feats = dataset.pixValue_img.to("cuda:0", non_blocking=True)
                    normed_feats = F.normalize(normed_feats)
                    all_nns_max = []
                    all_nns_min = []
                    step = normed_feats.shape[0] // n_batches
                    print(normed_feats.shape)
                    for i in tqdm(range(0, normed_feats.shape[0], step)):
                        # torch.cuda.empty_cache()
                        batch_feats = normed_feats[i:i + step, :]
                        pairwise_sims = torch.einsum("nf,mf->nm", batch_feats, normed_feats)
                        all_nns_max.append(torch.topk(pairwise_sims, 7, largest=True, sorted=False)[1])
                        all_nns_min.append(torch.topk(pairwise_sims, 25, largest=False, sorted=False)[1])
                        # del pairwise_sims
                    nearest_neighbors = torch.cat(all_nns_max, dim=0)
                    farthest_element = torch.cat(all_nns_min, dim=0)
                    np.savez_compressed(nearest_feature_cache_file, nns=nearest_neighbors.to("cpu").numpy())
                    np.savez_compressed(farthest_feature_cache_file, nns=farthest_element.to("cpu").numpy())
                    print("Saved NNs", cfg.model_type, nice_dataset_name, image_set)


if __name__ == "__main__":
    prep_args()
    my_app()
