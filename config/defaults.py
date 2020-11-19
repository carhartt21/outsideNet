from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = "weights/outside15k-resnet50-outsideNet"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.root_dataset = "./data/"
_C.DATASET.list_train = "./data/training.odgt"
_C.DATASET.list_val = "./data/validation.odgt"
_C.DATASET.list_stats = ""
_C.DATASET.num_class = 150
# multiscale train/test, size of short edge (int or tuple)
_C.DATASET.imgSizes = (300, 375, 450, 525, 600)
# maximum input image size of long edge
_C.DATASET.imgMaxSize = 1000
# maxmimum downsampling rate of the network
_C.DATASET.padding_constant = 8
# downsampling rate of the segmentation label
_C.DATASET.segm_downsampling_rate = 8
# randomly horizontally flip images when train/test
_C.DATASET.random_flip = True
# randomly crop images if the size exceeds the maximum size
_C.DATASET.random_crop = True
# info file
_C.DATASET.classInfo = "./data/outside15k.json"

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# architecture of net_encoder
_C.MODEL.arch_encoder = "resnet50"
# architecture of net_decoder
_C.MODEL.arch_decoder = "outsideNet"
# weights to finetune net_encoder
_C.MODEL.weights_encoder = ""
# weights to finetune net_decoder
_C.MODEL.weights_decoder = ""
# number of feature channels between encoder and decoder
_C.MODEL.fc_dim = 2048
# use spatial mask
_C.MODEL.spatial_mask = False
# align corners during upsampling
_C.MODEL.align_corners = False

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.batch_size_per_gpu = 2
# epochs to train for
_C.TRAIN.num_epoch = 20
# epoch to start training. useful if continuing from checkpoint
_C.TRAIN.start_epoch = 0
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.epoch_iters = 5000

_C.TRAIN.optim = "SGD"
_C.TRAIN.lr_encoder = 0.02
_C.TRAIN.lr_decoder = 0.02
# power in poly to drop LR
_C.TRAIN.lr_pow = 0.9
# momentum for sgd, beta1 for adam
_C.TRAIN.beta1 = 0.9
# weights regularizer
_C.TRAIN.weight_decay = 1e-4
# the weighting of deep supervision loss
_C.TRAIN.deep_sup_scale = 0.4
# fix bn params, only under finetuning
_C.TRAIN.fix_bn = False
# number of data loading workers
_C.TRAIN.workers = 16
# frequency to display
_C.TRAIN.disp_iter = 20
# use validation during training
_C.TRAIN.eval = False
# step size for validation calculation
_C.TRAIN.eval_step = 5
# best score from previous Training
_C.TRAIN.best_score = 0
# optimizer data from previous training
_C.TRAIN.optim_data = ""

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# currently only supports 1
_C.VAL.batch_size = 1
# output visualization during validation
_C.VAL.visualize = False
# the checkpoint to evaluate on
_C.VAL.checkpoint = "epoch_20.pth"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# currently only supports 1
_C.TEST.batch_size = 1
# the checkpoint to test on
_C.TEST.checkpoint = "epoch_20.pth"
# folder to output visualization results
_C.TEST.result = "./"
