import os
import os.path as osp
from os.path import dirname, abspath

DEFAULT_SEED = 42 #seed
DS_SEED = 123  # uses this seed when splitting datasets

# -------------- Paths
CONFIG_PATH = abspath(__file__)
SRC_ROOT = dirname(CONFIG_PATH)
PROJECT_ROOT = dirname(SRC_ROOT)
CACHE_ROOT = osp.join(SRC_ROOT, 'cache')
DATASET_ROOT = osp.join(PROJECT_ROOT, 'data')
DEBUG_ROOT = osp.join(PROJECT_ROOT, 'debug')
MODEL_DIR = osp.join(PROJECT_ROOT, 'models')
VICTIM_DIR = osp.join(MODEL_DIR, 'victim\\cifar-WidResNet28-10')
TRANSMI_DIR = osp.join(MODEL_DIR, 'shadow\\cifar100-WideResNet28-2') # used in transfer MI
PRETRAIN_DIR = None

attack_model_dir = osp.join(MODEL_DIR, "adversary\\ADV_DIR")
transfer_set_out_dir = osp.join(MODEL_DIR, "adversary\\TRANSFER_SET")
shadow_model_dir = osp.join(MODEL_DIR, "shadow")

# -------------- URLs
ZOO_URL = 'http://datasets.d2.mpi-inf.mpg.de/blackboxchallenge'

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]

batch_size = 32
epoch = 3 #150
use_default_initial = False
initial_seed = 200                                              # number of initial seed
k = 200
num_iter = 1

num_classes = 10

read_shadow_from_path = False                                   # resume shadow model
read_attack_mia_model_from_path = True                          # resume shadow-model MI attack model
mi_test = True                                                  # if having test process in shadow-model MI attack model training.
shadow_queryset = 'CIFAR10-10000,DownSampleImagenet32-10000'    #
start_shadow_test = 0                                           #
start_shadow_test_out = 10000                                   #
transfer_mi = False                                             # metric-based shadow-model MI training method(1)
augment_mi = False                                              # metric-based shadow-model MI training method(2)

unsuper_data_start = 100000                                     # start index of non-members in unsupervised MI
unsuper_data = 1000                                             # size of non-members selected in unsupervised MI threshold decsion.

imp_vic_mem = True                                              # True == Pre-Filter is on
vic_mem_method = 'shadow_model'                                 # 'shadow_model' 'unsupervised' the MI module of Pre-Filter

test_dataset = 'CIFAR10'                                        # model extraction test dataset; 'MNIST', 'CIFAR1O','SVHN'
attack_model_arch = 'wide_resnet28_10'                          # 'resnet18', wrn_28_2
victim_model_arch = 'wide_resnet28_10'
queryset ='CIFAR10-500,CIFAR100-0,DownSampleImagenet32-1000'
trainingbound = 500                                             # the dividing index between training and non-training data

seed = 1337                                                     # used in random, np.random

log_interval = 5