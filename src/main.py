import os
import register
import utils
import world
import torch
import time
import Procedure
from os.path import join
from register import dataset
from typing import Optional
from tensorboardX import SummaryWriter

utils.set_seed(world.seed)
print(">>SEED:", world.seed)

if not os.path.exists('logs'):
    os.mkdir('logs')

if not os.path.exists('embs'):
    os.mkdir('embs')

config = f'{world.args.dataset}_seed{world.args.seed}_{world.args.model}_dim{world.args.recdim}_lr{world.args.lr}_dec{world.args.decay}_alpha{world.args.alpha}_beta{world.args.beta}_gamma{world.args.gamma}'

if world.args.model == 'lgn':
    config += f'_nl{world.args.layer}'

log_path = f'logs/{config}.txt'
emb_path = f'embs/{config}'

if os.path.exists(emb_path):
    print('Exists.')
    exit(0)
else:
    os.mkdir(emb_path)

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")

if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

Neg_k = 1
w: Optional[SummaryWriter] = None

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + str(world.comment)))
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()

        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')
        # save model
        torch.save(Recmodel.state_dict(), join(emb_path, f'{epoch}.pth'))
finally:
    if world.tensorboard:
        w.close()