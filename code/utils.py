import pickle
from collections import Counter
from character_dataset import CharacterDataset
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from constants import *
DL_ARGS = {'batch_size': 64, 'shuffle': True }

def save_struct(filename, struct):
    file = open(filename,'wb')
    pickle.dump(struct,file)
    file.close()

def load_struct(filename):
    file = open(filename,'rb')
    struct = pickle.load(file)
    file.close()
    return struct

def iteration_number(epoch, loader, batch_idx):
    """ calculate iteration number """
    return (epoch * len(loader)) + batch_idx

def build_embedding_set(dataset, categories, size, exclude_ids=[]):
    xs = []
    ys = []
    # we filter out categories because this turkey takes forever:
    valid_indexes = [idx for idx,rec in enumerate(dataset) if rec['category'] in categories]

    # we further filter (without replacement):
    valid_indexes = [idx for idx in valid_indexes if idx not in exclude_ids]

    indexes = np.random.choice(valid_indexes, size, replace=False)
    for idx in indexes:
        rec = dataset[idx]
        x = rec['abstract']
        y = rec['category']
        xs.append(x)
        ys.append(y)
    return xs, ys, indexes

def data_from_pickle():
    dbpedia = load_struct('data/train_test.pickle')
    counts = Counter([ rec['category'] for rec in dbpedia['train']])
    categories = list(counts.keys())
    return dbpedia, categories

def build_test_loader(test_categories, size):
    data, categories = data_from_pickle()
    xs_test, y_test, indexes = build_embedding_set(data['test'], test_categories, size)
    dataset_test = CharacterDataset(xs_test, y_test, all_labels=CATEGORIES)
    return torch.utils.data.DataLoader(dataset=dataset_test, **DL_ARGS), indexes

def build_train_loader(train_categories, size, exclude_ids=[]):
    data, categories = data_from_pickle()
    xs_train, y_train, indexes = build_embedding_set(data['train'], train_categories, size, exclude_ids)
    dataset_train = CharacterDataset(xs_train, y_train, all_labels=CATEGORIES)
    return torch.utils.data.DataLoader(dataset=dataset_train, **DL_ARGS), indexes


def train_epoch(model, train_loader, optimizer, epoch):
    """ Train the model """
    embedding_log = 20
    total_loss = 0
    total_size = 0
    print("Epoch: {}".format(epoch))
    criterion = nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.data[0]
        total_size += data.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss / total_size))



def valid_epoch(model, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


