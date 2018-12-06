import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset


EMBEDDING_MAP = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
def preprocess_abstract(text, mat_width=1014):
    num_rows = len(EMBEDDING_MAP)
    embedded = np.zeros((num_rows, mat_width))
    abstract_embedded = []
    cur_col = 0
    for char in text.lower():
        if char in EMBEDDING_MAP:
            if cur_col >= mat_width:
                continue
            embedded[EMBEDDING_MAP.index(char)][cur_col]= 1
            cur_col += 1


    return embedded

"""
Our custom dataloader
"""
class CharacterDataset(Dataset):
    def __init__(self, data, labels=[], transform=None, all_labels=[]):
        self.data = data
        self.labels = np.asarray([all_labels.index(lbl) for lbl in labels])
        self.transform = transform

    def __getitem__(self, index):
        single_label = 999
        if len(self.labels) > 0:
            single_label = self.labels[index]

        result = preprocess_abstract(self.data[index])
        #result = self.transform(abstract)
        return (torch.from_numpy(result).float(), single_label)


    def __len__(self):
        return len(self.data)
