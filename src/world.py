import os
import torch
from os.path import join
from parse import parse_args
import multiprocessing

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, 'src')
FILE_PATH = join(CODE_PATH, 'checkpoints')

args = parse_args()
seed = args.seed

all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book', 'ml-100k', 'ml-1m', 'yelp2018-ass']
dataset = args.dataset
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")

all_models  = ['mf', 'lgn']
model_name = args.model
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")

GPU = torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if GPU else "cpu")

config = {}
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lr'] = args.lr
config['decay'] = args.decay
config['lightGCN_n_layers']= args.layer
config['A_n_fold'] = args.a_fold
config['A_split'] = False
config['pretrain'] = args.pretrain
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['alpha'] = args.alpha
config['beta'] = args.beta
config['gamma'] = args.gamma

def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

LOAD = args.load
tensorboard = args.tensorboard
comment = args.comment
TRAIN_epochs = args.epochs
topks = eval(args.topks)
PATH = args.path
CORES = multiprocessing.cpu_count()
BOARD_PATH = join(CODE_PATH, 'runs')