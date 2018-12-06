import pickle
from collections import Counter
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import numpy as np

from character_dataset import CharacterDataset
#from simple_net import SimpleNet
from char_cnn import CharacterLevelCNN
from datetime import datetime


def save_struct(filename, struct):
    file = open(filename,'wb')
    pickle.dump(struct,file)
    file.close()

def load_struct(filename):
    file = open(filename,'rb')
    struct = pickle.load(file)
    file.close()
    return struct



TRAIN_PER_CAT = 40_000
TEST_PER_CAT = 5_000
def train_test_split():
    records_from_dbpedia = load_struct('data/dbpedia_dump.pickle')
    counts = Counter([ rec['category'] for rec in records_from_dbpedia])
    categories = list(counts.keys())

    test_dataset = []
    train_dataset = []
    for category in categories:
        subset = [ rec for rec in records_from_dbpedia if rec['category'] == category]
        indexes = set(np.arange(0, len(subset)))
        train_indexes = np.random.choice(list(indexes), TRAIN_PER_CAT, replace=False)
        test_indexes = np.random.choice(list(indexes - set(train_indexes)), TEST_PER_CAT, replace=False)
        for idx in train_indexes:
            train_dataset.append(subset[idx])
        for idx in test_indexes:
            test_dataset.append(subset[idx])
    return train_dataset, test_dataset

def pickle_train_test():
    train, test = train_test_split()
    save_struct('data/train_test.pickle', { 'train': train, 'test': test })


np.set_printoptions(threshold=np.nan)


def build_embedding_set(dataset, categories):
    valid_categories = list(['Company', 'EducationalInstitution', 'Artist'])
    xs = []
    ys = []
    # we filter out to four categories because this turkey takes forever:
    category_count = {'Company': 0, 'EducationalInstitution': 0, 'Artist': 0}
    for idx, rec in enumerate(dataset):
        if rec['category'] in valid_categories:
            x = rec['abstract']
            y = rec['category']
            if category_count[y] > 3000:
                continue
            category_count[y] += 1
            xs.append(x)
            ys.append(y)
    return xs, ys


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




if __name__ == '__main__':
    train_test_dbpedia = load_struct('data/train_test.pickle')
    counts = Counter([ rec['category'] for rec in train_test_dbpedia['train']])
    # get label embeddings:
    categories = list(counts.keys())

    xs_train, y_train = build_embedding_set(train_test_dbpedia['train'], categories)
    xs_test, y_test = build_embedding_set(train_test_dbpedia['test'], categories)

    train_transform = transforms.Compose([ transforms.ToTensor(), ])
    test_transform = transforms.Compose([ transforms.ToTensor(), ])

    dataset_train = CharacterDataset(xs_train, y_train, all_labels=categories, transform=train_transform)
    dataset_test = CharacterDataset(xs_test, y_test, all_labels=categories, transform=test_transform)

    dl_args = {'batch_size': 128, 'shuffle': True, 'num_workers': 1 }
    learning_rate = 0.001
    epochs =  10

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, **dl_args)
    valid_loader = torch.utils.data.DataLoader(dataset=dataset_test, **dl_args)

    model = CharacterLevelCNN()
    model = model.cuda()
    """ train and evaluate model performance """
    torch.manual_seed(1)
    time = datetime.now().strftime("%I_%M%S_{}_".format("SimpleNet"))
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, optimizer, epoch)
        valid_epoch(model, valid_loader, epoch)
