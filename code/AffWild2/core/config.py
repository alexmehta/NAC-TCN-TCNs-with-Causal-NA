"""
Original source: https://github.com/facebookresearch/pycls/blob/master/pycls/core/config.py
Latest commit 2c152a6 on May 6, 2021
"""

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""


# Global config object (example usage: from core.config import cfg)
import argparse
import os
import sys
from .io import cache_url, pathmgr
from yacs.config import CfgNode
_C = CfgNode()
cfg = _C

# ---------------------------------- Model options ----------------------------------- #
_C.MODEL = CfgNode()

# Model backbone
_C.MODEL.BACKBONE = 'regnet'

# Freeze backbone or not
_C.MODEL.BACKBONE_FREEZE = []

# Freeze Batchnorm
_C.MODEL.FREEZE_BATCHNORM = True

# BACKBONE PRETRAINED
_C.MODEL.BACKBONE_PRETRAINED = ''

# Number of classes
_C.MODEL.NUM_CLASSES = 15

# Loss function (see pycls/models/loss.py for options)
_C.MODEL.LOSS_FUN = "mse"

# Number of hidden units in last layers
_C.MODEL.FC_HIDDEN = 32

# Temporal model type
_C.MODEL.TEMPORAL_TYPE = 'tcn'  # tcn or lstm

# Use position or not
_C.MODEL.USE_POSITION = True

# List of pre-trained features
_C.MODEL.FEATURES = ['effb0']
# ------------------------------- TCN options ------------------------------- #
_C.TCN = CfgNode()

_C.TCN.NUM_CHANNELS = 512
# TCN channels
_C.TCN.NUM_STACK = 2

# TCN Dilations
_C.TCN.DILATIONS = 4

# TCN Kernel size
_C.TCN.K_SIZE = 3

# TCN Dropout
_C.TCN.DROPOUT = 0.

# Use WeightNorm in TCN or not
_C.TCN.NORM = True

# Number of temporal module (head)
_C.TCN.NUM_HEAD = 1

# ------------------------------- TRANSFORMER options ------------------------------- #
_C.TRANF = CfgNode()
# Number of encoder, decoder
_C.TRANF.NUM_ENC_DEC = 3

# Number of head
_C.TRANF.NHEAD = 8

# Number DIM of FEEDFORWARD
_C.TRANF.DIM_FC = 1024

# Transformer dropout
_C.TRANF.DROPOUT = 0.3

# ------------------------------- LSTM options ------------------------------- #
_C.LSTM = CfgNode()

# LSTM HIDDEN_SIZE
_C.LSTM.HIDDEN_SIZE = 64

# LSTM Num layers
_C.LSTM.NUM_LAYERS = 4

# LSTM Bidirectional or not
_C.LSTM.BIDIREC = False

# LSTM Dropout
_C.LSTM.DROPOUT = 0.

# ------------------------------- GRU options ------------------------------- #
_C.GRU = CfgNode()

# LSTM HIDDEN_SIZE
_C.GRU.HIDDEN_SIZE = 256

# LSTM Num layers
_C.GRU.NUM_LAYERS = 2

_C.GRU.BIDIRECTIONAL = False

# GRU Dropout
_C.GRU.DROPOUT = 0.

# -------------------------------- Optimizer options --------------------------------- #
_C.OPTIM = CfgNode()

_C.OPTIM.NAME = 'adam'
# Learning rate ranges from BASE_LR to MIN_LR*BASE_LR according to the LR_POLICY
_C.OPTIM.BASE_LR = 0.1
_C.OPTIM.MIN_LR = 0.0

# Learning rate policy select from {'cos', 'exp', 'lin', 'steps'}
_C.OPTIM.LR_POLICY = "cos"

# Steps for 'steps' policy (in epochs)
_C.OPTIM.STEPS = []

# Learning rate multiplier for 'steps' policy
_C.OPTIM.LR_MULT = 0.1

# Maximal number of epochs
_C.OPTIM.MAX_EPOCH = 200

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WEIGHT_DECAY = 5e-4

# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.OPTIM.WARMUP_FACTOR = 0.1

# Gradually warm up the OPTIM.BASE_LR over this number of epochs
_C.OPTIM.WARMUP_EPOCHS = 0

# Exponential Moving Average (EMA) update value
_C.OPTIM.EMA_ALPHA = 1e-5

# Iteration frequency with which to update EMA weights
_C.OPTIM.EMA_UPDATE_PERIOD = 32

# Use swa or not
_C.OPTIM.USE_SWA = True
# Do Test only, specify test weight
_C.TEST_ONLY = "none"
# Focal loss
_C.OPTIM.FOCAL_ALPHA = 0.25
_C.OPTIM.FOCAL_GAMMA = 2.0
# --------------------------------- Training options --------------------------------- #
_C.TRAIN = CfgNode()

# Dataset and split
_C.TRAIN.DATASET = ""
_C.TRAIN.SPLIT = "train"

# Total mini-batch size
_C.TRAIN.BATCH_SIZE = 1

# If True train using mixed precision
_C.TRAIN.MIXED_PRECISION = False

# Accumulated gradients runs K small batches of size N before doing a backwards pass
_C.TRAIN.ACCUM_GRAD_BATCHES = 1

# Label smoothing
_C.TRAIN.LABEL_SMOOTHING = 0.1

# Weighted loss function
_C.TRAIN.LOSS_WEIGHTS = True

# Resume training from the latest checkpoint in the output directory
_C.TRAIN.AUTO_RESUME = True

# Limit train batches
_C.TRAIN.LIMIT_TRAIN_BATCHES = 1.

_C.TRAIN.DROP_PERC = 0.3

# Weights to start training from
_C.TRAIN.WEIGHTS = ""
# --------------------------------- Testing options ---------------------------------- #
_C.TEST = CfgNode()

# Dataset and split
_C.TEST.DATASET = ""
_C.TEST.SPLIT = "val"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 1

# Weights to use for testing
_C.TEST.WEIGHTS = ""

# ------------------------------- Data loader options -------------------------------- #
_C.DATA_LOADER = CfgNode()

# Modify expr labels, convert to multi-label problem
_C.DATA_LOADER.EXPR_MLB = False

# Number of data loader workers per process
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = False

# ROOT of DATASET
_C.DATA_LOADER.DATA_DIR = '/home/alex/abaw'
_C.DATA_LOADER.KFOLD_DATA_DIR = '/mnt/XProject/dataset'

_C.DATA_LOADER.EMO_INDEX = -1

# Sequence length
_C.DATA_LOADER.SEQ_LEN = 128

# Image size
_C.DATA_LOADER.IMG_SIZE = 224

# Sampling method to split video into seqs
# random => random select SEQ_LEN elements from a video
# sequentially => select continuous frame to form a sequence
_C.DATA_LOADER.SAMPLING_METHOD = 'sequentially'

_C.DATA_NAME = 'eev'
# ---------------------------------- CUDNN options ----------------------------------- #
_C.CUDNN = CfgNode()

# Perform benchmarking to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# ----------------------------------- Misc options ----------------------------------- #
# Optional description of a config
_C.DESC = ""

# If True output additional info to log
_C.VERBOSE = True

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1

_C.TASK = 'VA'
# Output directory
_C.OUT_DIR = "./tmp"

# Config destination (in OUT_DIR)
_C.CFG_DEST = "config.yaml"

# Note that non-determinism is still be present due to non-deterministic GPU ops
_C.RNG_SEED = 1

# Log destination ('stdout' or 'file')
_C.LOG_DEST = "stdout"

# Log period in iters
_C.LOG_PERIOD = 10

# Logger (wandb or TensorBoard)
_C.LOGGER = "TensorBoard"

# Notes
_C.NOTE = ""

# Model type
_C.MODEL_TYPE = 'gru'

# Combine loss
_C.COMBINE_LOSS = False

# Kfold
_C.N_KFOLD = -1

# Add feature
_C.ADD_FEAT = '400mf'

# local attention
_C.LOCAL_ATT = 32
# T_mult
_C.T_MULT = 2

# Value combine loss
_C.VALUE_COMBINE_LOSS = 0.2

# Models weights referred to by URL are downloaded to this local cache
_C.DOWNLOAD_CACHE = "/tmp/pycls-download-cache"

# Fast dev run, > 0 run fast dev only for check training/validation logic
_C.FAST_DEV_RUN = 0
# ---------------------------------- Default config ---------------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_and_infer_cfg(cache_urls=True):
    """Checks config values invariants."""
    err_str = "The first lr step must start at 0"
    assert not _C.OPTIM.STEPS or _C.OPTIM.STEPS[0] == 0, err_str
    data_splits = ["train", "val", "test"]
    err_str = "Data split '{}' not supported"
    assert _C.TRAIN.SPLIT in data_splits, err_str.format(_C.TRAIN.SPLIT)
    assert _C.TEST.SPLIT in data_splits, err_str.format(_C.TEST.SPLIT)
    err_str = "Mini-batch size should be a multiple of NUM_GPUS."
    assert _C.TRAIN.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    assert _C.TEST.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)
    if cache_urls:
        cache_cfg_urls()


def cache_cfg_urls():
    """Download URLs in config, cache them, and rewrite cfg to use cached file."""
    _C.TRAIN.WEIGHTS = cache_url(_C.TRAIN.WEIGHTS, _C.DOWNLOAD_CACHE)
    _C.TEST.WEIGHTS = cache_url(_C.TEST.WEIGHTS, _C.DOWNLOAD_CACHE)


def dump_cfg(out_dir=''):
    """Dumps the config to the output directory."""
    if out_dir == '':
        out_dir = _C.OUT_DIR
    cfg_file = os.path.join(out_dir, _C.CFG_DEST)
    with pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)
    return cfg_file


def load_cfg(cfg_file):
    """Loads config from specified file."""
    with pathmgr.open(cfg_file, "r") as f:
        _C.merge_from_other_cfg(_C.load_cfg(f))


def reset_cfg():
    """Reset config to initial state."""
    _C.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_fom_args(description="Config file options."):
    """Load config from command line arguments and set any specified options."""
    parser = argparse.ArgumentParser(description=description)
    help_s = "Config file location"
    parser.add_argument("--cfg", dest="cfg_file",
                        help=help_s, required=True, type=str)
    help_s = "See pycls/core/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None,
                        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    load_cfg(args.cfg_file)
    _C.merge_from_list(args.opts)
