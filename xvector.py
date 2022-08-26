# 2022/08/24 15:37
# have a good day!
import torch
from torch import nn
import librosa
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os

BATCH_SIZE = 30
class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        list = os.listdir(self.root_dir + '/' + self.label_dir)
        new = []
        for each in list:
            if each[0] == '.':
                pass
            else:
                new.append(each)
        self.audio_list = new
        self.readpath_list = [self.root_dir + '/' + self.label_dir + '/' + each for each in self.audio_list]

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, item):
        y, sr = librosa.load(self.readpath_list[item], sr=16000)
        y = y[0:30000]
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30, dct_type=2)
        x = torch.tensor(mfccs)
        if 'zhao' in self.label_dir:
            return x, 0
        else:
            return x, 1

dataset1 = MyDataset('audio_dataset', 'zhao')
dataset2 = MyDataset('audio_dataset', 'lian')
lian_testdataset = MyDataset('audio_dataset', 'lian_test')
zhao_testdataset = MyDataset('audio_dataset', 'zhao_test')
testdataset = lian_testdataset + zhao_testdataset
test_loader = DataLoader(testdataset, batch_size=32, shuffle=True)
dataset = dataset1 + dataset2
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

class xvecTDNN(nn.Module):
    def __init__(self, numSpkrs, p_dropout):
        super(xvecTDNN, self).__init__()
        self.tdnn1 = nn.Conv1d(in_channels=30, out_channels=512, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn_tdnn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn_tdnn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(3000,512)
        self.bn_fc1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        self.fc2 = nn.Linear(512,512)
        self.bn_fc2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc2 = nn.Dropout(p=p_dropout)

        self.fc3 = nn.Linear(512,numSpkrs)

    def forward(self, x, eps):
        # Note: x must be (batch_size, feat_dim, chunk_len)
        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
        x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))

        if self.training:
            shape = x.size()
            noise = torch.FloatTensor(shape)
            torch.randn(shape, out=noise)
            x += noise*eps

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        # x = self.dropout_fc1(F.relu(self.fc1(stats)))
        # x = self.dropout_fc2(F.relu(self.fc2(x)))
        x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(stats))))
        x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x

# # plt.imshow(mfccs)
net = xvecTDNN(2, 0.2)
print(net)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X, 0.01)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 90 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X, 0.01)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epoch = 500
for i in range(epoch):
    train(loader, net, loss_fn, optimizer)
    test(test_loader, net, loss_fn)




