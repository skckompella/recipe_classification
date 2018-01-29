import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import cPickle as pkl
import numpy as np

import constants

class RecipeDataset(Dataset):
    """
    Uses Pytorch Dataset class to handle batching
    """
    def __init__(self, x_train_path, y_train_path, x_dev_path, y_dev_path):
        """
        All parameters below are file paths
        :param x_train_path:
        :param y_train_path:
        :param x_dev_path:
        :param y_dev_path:
        """
        with open(x_train_path, "rb") as fp:
            self.x_train = pkl.load(fp)
        len = self.x_train.shape[0]
        self.width = self.x_train.shape[1]
        shuffled_indices = np.random.randint(0, len-1, size=len-1)
        self.x_train = self.x_train[shuffled_indices]
        with open(y_train_path, "rb") as fp:
            self.y_train = pkl.load(fp)
        self.y_train = self.y_train[shuffled_indices]
        self.train_len = len -1

        with open(x_dev_path, "rb") as fp:
            self.x_dev = pkl.load(fp)
        with open(y_dev_path, "rb") as fp:
            self.y_dev = pkl.load(fp)
        #Possible improvement - split dev set

    def __len__(self):
        return self.train_len

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

    def get_dev_set(self):
        return torch.from_numpy(self.x_dev), torch.from_numpy(self.y_dev)

    def get_max_len(self):
        return self.width

class RecipeNet(nn.Module):
    """
    k-convolutional layers (in parallel) with different filter sizes;
    followed by a few fully connected layer and a softmax layer
    """
    def __init__(self, vocab_len, num_labels, max_len):
        """
        :param vocab_len: Vocabulary length
        :param num_labels: Number of labels
        :param max_len: Max length of recipe in the train set
        """

        super(RecipeNet, self).__init__()
        NULL_IDX = 0
        self.vocab_len = vocab_len
        self.num_labels = num_labels
        self.embedding_size = constants.EMBEDDING_SIZE
        self.kernel_sizes = constants.KERNEL_SIZES

        self.embedding = nn.Embedding(vocab_len, self.embedding_size, padding_idx=NULL_IDX)
        self.convs = nn.ModuleList([nn.Conv1d(self.embedding_size, self.embedding_size, K)
                                    for K in self.kernel_sizes])
        self.linear = nn.Linear(len(self.kernel_sizes) * self.embedding_size, num_labels)
        self.fc1 = nn.Linear(max_len*self.embedding_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, num_labels)
        self.dropout = nn.Dropout(constants.DROPOUT)
        self.softmax = nn.LogSoftmax()


    def forward(self, x):
        """
        :param x: input
        :return: Softmax scores
        """
        x = self.embedding(x)
        x = x.permute(0,2,1)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.linear(x)

        scores = self.softmax(logit)
        return scores



class RecipeNet2(nn.Module):
    def __init__(self, vocab_len, num_labels, max_len):
        """
        :param vocab_len: Vocabulary length
        :param num_labels: Number of labels
        :param max_len: Max length of recipe in the train set
        """

        super(RecipeNet2, self).__init__()
        NULL_IDX = 0
        self.vocab_len = vocab_len
        self.num_labels = num_labels
        self.embedding_size = constants.EMBEDDING_SIZE
        self.kernel_sizes = constants.KERNEL_SIZES

        self.embedding = nn.Embedding(vocab_len, self.embedding_size, padding_idx=NULL_IDX)
        self.fc1 = nn.Linear(max_len*self.embedding_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, num_labels)
        self.dropout = nn.Dropout(constants.DROPOUT)
        self.softmax = nn.LogSoftmax()


    def forward(self, x):
        """
        :param x: input
        :return: Softmax scores
        """
        x = self.embedding(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        logit = self.fc3(x)

        scores = self.softmax(logit)
        return scores


