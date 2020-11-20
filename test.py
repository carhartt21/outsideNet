# System libs
import os
import argparse
from distutils.version import LooseVersion
import json
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
import csv
import logging
# Our libs
from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from model import outsideNet
from utils import colorEncode, find_recursive, setup_logger
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from config import cfg
from models.modelsummary import get_model_summary

def visualize_result(data, pred, cfg):
    colors = []
    names = {}
    with open(cfg.DATASET.classInfo) as f:
        cls_info = json.load(f)
    for c in cls_info:
        names[c] = cls_info[c]['name']
        colors.append(cls_info[c]['color'])
    colors = np.array(colors, dtype='uint8')
    (img, info) = data
    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names[str(uniques[idx] + 1)]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(
        os.path.join(cfg.TEST.result, '{}_{}{}.png'
                     .format(img_name[:-4], cfg.MODEL.arch_encoder, cfg.MODEL.arch_decoder)))

def test(segmentation_module, loader, gpu):
    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        label_size = (batch_data['img_ori'].shape[0],
                    batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, label_size[0], label_size[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)
                # forward pass
                pred_tmp = segmentation_module(feed_dict, label_size=label_size)
                scores += pred_tmp / len(cfg.DATASET.imgSizes)
            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        # visualization
        visualize_result((batch_data['img_ori'], batch_data['info']), pred, cfg)
        pbar.update(1)


def main(cfg, gpu):
    torch.cuda.set_device(gpu)
    dump_model = False

    crit = nn.NLLLoss(ignore_index=-1)
    network = outsideNet(
        crit,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights_resNet=cfg.MODEL.weights_encoder,
        weights=cfg.MODEL.weights_decoder,
        spatial_mask=cfg.MODEL.spatial_mask,
        use_softmax=True
    )

    # Dataset and Loader
    dataset_test = TestDataset(
        cfg.list_test,
        cfg.DATASET, 
        spatial_mask=cfg.MODEL.spatial_mask)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    network.cuda()

    if dump_model:
        dump_input = torch.rand((1, 3, 1920, 1080))
        with open('dump_model.txt', 'w') as file:
            file.write(get_model_summary(network.cuda(), dump_input.cuda(), verbose=True))

    # Main loop
    test(network, loader_test, gpu)

    print('Inference done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
    parser.add_argument(
        "--imgs",
        required=True,
        type=str,
        help="an image paths, or a directory name"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu id for evaluation"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # generate testing image list
    if os.path.isdir(args.imgs):
        print(args.imgs)
        imgs = find_recursive(args.imgs)
    else:
        imgs = [args.imgs]
    assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    cfg.list_test = [{'fpath_img': x} for x in imgs]

    if not os.path.isdir(cfg.TEST.result):
        os.makedirs(cfg.TEST.result)

    main(cfg, args.gpu)
