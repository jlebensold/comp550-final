import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Worker:
    def __init__(self, name, model, optimizer):
        self.name = name
        self.optimizer = optimizer
        self.model = model

    def train_communication_round(self, train_loader, epoch):
        total_loss = 0
        total_size = 0
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data).cuda(), Variable(target).cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item())
                )


    def valid_epoch(self, test_loader, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                output = self.model(data)
                test_loss += criterion(output, target).data[0]
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\n {}: Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            self.name,
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
