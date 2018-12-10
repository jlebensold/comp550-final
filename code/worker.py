import matplotlib.pyplot as plt
from torchviz import make_dot, make_dot_from_trace
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import iteration_number
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from datetime import datetime

class Worker:
    def __init__(self, name, model, optimizer, experiment="Undefined"):
        self.name = name
        self.optimizer = optimizer
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.current_model_weights = {}
        time = datetime.now().strftime("%I_%M%S_{}".format(experiment))
        self.writer = SummaryWriter('../training_logs/{}/{}'.format(self.name, time))

    def train_communication_round(self, train_loader, comm_round):
        self.store_model_weights()
        total_loss = 0
        total_size = 0
        criterion = nn.CrossEntropyLoss()

        self.model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)

            loss = criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('train/loss', loss.data.item(), 
                    iteration_number(comm_round, train_loader, batch_idx))
            if batch_idx % 10 == 0:
                print('Train Comm Round: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    comm_round,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item())
                )
        self.update_params_delta()

    def store_model_weights(self):
        for name, param in self.model.named_parameters():
            # TODO: check if this constraint is required:
            if param.requires_grad:
                self.current_model_weights[name] = param.data.detach()


    def update_params_delta(self):
        """ subtract the current model parameters from the updated model parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.current_model_weights.keys():
                   self.current_model_weights[name] =  param.data - self.current_model_weights[name]


    def valid_comm_round(self, test_loader, comm_round):
        self.model.eval()
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target) 
                test_loss += loss.item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum().item()
                self.writer.add_scalar('valid/loss', loss.data.item(),           
                        iteration_number(comm_round, test_loader, idx))

        test_loss /= len(test_loader.dataset)

        acc = 100. * correct / len(test_loader.dataset)
        self.writer.add_scalar('valid/accuracy', acc, (comm_round * len(test_loader)))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), acc))
