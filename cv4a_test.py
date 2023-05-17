import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch.optim import Adam, SGD, lr_scheduler
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from cv4a_data import get_cv4a_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score

import pdb


class cv4aDataset(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.images = images
        self.img_labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label.to(torch.long)


class CnnNica(nn.Module):

    def __init__(self, N, num_classes):
        super(CnnNica, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(N, out_channels=12, kernel_size=(3, 5, 5),
                               padding=(1, 0, 0), bias=False)
        self.conv2 = nn.Conv3d(12, out_channels=36, kernel_size=(3, 5, 5),
                               padding=(1, 0, 0), bias=False)
        self.dropout3d = nn.Dropout3d(0.2)
        self.dropout = nn.Dropout(0.5)
        self.bn3d_1 = nn.BatchNorm3d(12)
        self.bn3d_2 = nn.BatchNorm3d(36)
        self.bn1d_1 = nn.BatchNorm1d(512)
        self.bn1d_2 = nn.BatchNorm1d(128)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(36 * 2 * 5 * 5, 512)  # 5*5 from image dimension
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)


    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool3d(self.dropout3d(F.relu(self.bn3d_1(self.conv1(x)))),
                         (2, 2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool3d(self.dropout3d(F.relu(self.bn3d_2(self.conv2(x)))),
                         (2, 2, 2), stride=(1, 2, 2))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.bn1d_1(self.fc1(self.dropout(x))))
        x = F.relu(self.bn1d_2(self.fc2(self.dropout(x))))
        x = self.fc3(self.dropout(x))
        return x


class CnnOrig(nn.Module):

    def __init__(self, N, num_classes):
        super(CnnOrig, self).__init__()
        self.conv1 = nn.Conv3d(N, out_channels=32, kernel_size=(3, 5, 5),
                               padding=(1, 0, 0), bias=False)
        self.conv2 = nn.Conv3d(32, out_channels=64, kernel_size=(3, 5, 5),
                               padding=(1, 0, 0), bias=False)
        self.dropout3d = nn.Dropout3d(0.2)
        self.dropout = nn.Dropout(0.6)
        self.bn3d_1 = nn.BatchNorm3d(32)
        self.bn3d_2 = nn.BatchNorm3d(64)
        self.bn1d_1 = nn.BatchNorm1d(256)
        self.bn1d_2 = nn.BatchNorm1d(128)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(64 * 2 * 5 * 5, 256)  # 5*5 from image dimension
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)


    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool3d(self.dropout3d(F.relu(self.bn3d_1(self.conv1(x)))),
                         (2, 2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool3d(self.dropout3d(F.relu(self.bn3d_2(self.conv2(x)))),
                         (2, 2, 2), stride=(1, 2, 2))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.bn1d_1(self.fc1(self.dropout(x))))
        x = F.relu(self.bn1d_2(self.fc2(self.dropout(x))))
        x = self.fc3(self.dropout(x))
        return x


def train_epoch(model, train_loader, loss_fn, optimizer, device):
    size = len(train_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # make predictions and compute loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(model, test_loader, loss_fn, device):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def classification_test(features, labels, masks, nica_features=True):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    labels = torch.from_numpy(labels)
    masks = torch.from_numpy(masks)[:, None, :, :, :]
    features = torch.from_dlpack(features).to(torch.float32)
    num_classes = len(torch.unique(labels))
    n_data, N, D, H, W = features.shape

    # train/test splits
    batch_size = 16
    epochs = 100
    lr = 1e-3

    # stratified split sampling x-validation
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.15, random_state=0)
    fold = 0
    for train_index, val_index in sss.split(np.zeros(n_data), labels):
        print("***Fold: ", fold, "***")
        fold += 1
        # set up data loading
        tr_dataset = cv4aDataset(features[train_index],
                                 labels[train_index])
        te_dataset = cv4aDataset(features[val_index],
                                 labels[val_index])
        train_loader = DataLoader(tr_dataset, batch_size=batch_size,
                                  shuffle=True)
        test_loader = DataLoader(te_dataset, batch_size=batch_size)

        # set up model and optimizer
        #if nica_features:
        #    net = CnnNica(N, num_classes).to(device)
        #else:
        net = CnnOrig(N, num_classes).to(device)

        #and then real training -- unweighted
        loss = nn.CrossEntropyLoss()
        optimizer = Adam(net.parameters(), lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   10*len(train_loader))
        for t in range(epochs):
            print(f"!Epoch {t+1}\n-------------------------------")
            train_epoch(net, train_loader, loss, optimizer, device)
            scheduler.step()
            test(net, test_loader, loss, device)
            print("Done!")

    return 0



def test_rf(data, labels):
    n_data = len(labels)
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.15, random_state=0)
    fold = 0
    loss_list = []
    acc_list = []
    for train_index, val_index in sss.split(np.zeros(n_data), labels):
        fold += 1
        X_tr = data[train_index]
        y_tr = labels[train_index]
        X_te = data[val_index]
        y_te = labels[val_index]

        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_tr, y_tr)
        preds = rf.predict_proba(X_te)
        yhat = rf.predict(X_te)
        xe_loss = log_loss(y_te, preds)
        acc = accuracy_score(y_te, yhat)
        loss_list.append(xe_loss)
        acc_list.append(acc)

        print('--Fold ', fold,'--')
        print('X-entropy: ', xe_loss)
        print('Accuracy: ', acc)

    return loss_list, acc_list








#if __name__=="__main__":
#    T_t = 6
#    x, areas, field_masks, labels, dates = get_cv4a_data(args.cv4a_dir)
#    x = jnp.swapaxes(x, 1, 2)
#    x_tr = x[:, :, :T_t, :, :]
#    x_te = x[:, :, T_t:2*T_t, :, :]
#    num_data, M, _T_t, T_x, T_y = x_tr.shape
#    assert _T_t == T_t
#
#    # run baseline test
#    out = classification_test(x_tr, labels)
