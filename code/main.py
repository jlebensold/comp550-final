import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from char_cnn import CharacterLevelCNN
from datetime import datetime
from utils import build_train_loader, build_test_loader, valid_epoch, train_epoch
from federator import Federator
from worker import Worker
import time

from constants import *

WORKER_EXPERIMENTS = [
    [
        np.take(CATEGORIES,[0,1,2,3,4]),
        np.take(CATEGORIES,[0,1,2,3,4]),
        np.take(CATEGORIES,[0,1,2,3,4]),
    ],
    [
        np.take(CATEGORIES,[0,1,2]),
        np.take(CATEGORIES,[0,1,3]),
        np.take(CATEGORIES,[0,1,4]),
        np.take(CATEGORIES,[0,2,3]),
        np.take(CATEGORIES,[0,2,4]),
        np.take(CATEGORIES,[0,3,4]),
        np.take(CATEGORIES,[1,2,3]),
        np.take(CATEGORIES,[1,3,4]),
        np.take(CATEGORIES,[2,3,4]),
        np.take(CATEGORIES,[2,1,4]),
    ],
    [
        np.take(CATEGORIES,[0,1,2,3,4]),
        np.take(CATEGORIES,[0,1,2,3,4]),
        np.take(CATEGORIES,[0,1]),
        np.take(CATEGORIES,[2,3]),
    ],
    [
        np.take(CATEGORIES,[0,1]),
        np.take(CATEGORIES,[1,2]),
        np.take(CATEGORIES,[2,3]),
        np.take(CATEGORIES,[3,4]),
        np.take(CATEGORIES,[4,0]),
    ],

]

def optimizer_factory(model):
    return optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def model_factory():
    return torch.nn.DataParallel(CharacterLevelCNN(), [0,1,2,3]).cuda()

def new_worker(worker_data, idx, experiment):
    return Worker(name="W{}".format(idx), train_categories=worker_data[idx],
                  experiment=experiment, 
                  model_factory=model_factory, 
                  optimizer_factory=optimizer_factory)

def run_experiment(worker_data, worker_experiment_num):
    experiment = "w_exp={}".format(worker_experiment_num)

    workers = [new_worker(worker_data, wid, experiment) for wid in np.arange(0,len(worker_data))]

    federator = Federator(workers=workers, 
                         optimizer_factory=optimizer_factory, 
                         model_factory=model_factory,
                         experiment=experiment)
    federator.train_rounds()

def perform_all_experiments():
    for idx, worker_data in enumerate(WORKER_EXPERIMENTS):
        run_experiment(worker_data, idx)

def perform_federated_experiment(worker_experiment_num=None):
    worker_data = WORKER_EXPERIMENTS[int(worker_experiment_num)]
    run_experiment(worker_data, worker_experiment_num)

if __name__ == '__main__':
    fire.Fire()
