import os
import pickle
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score

torch.manual_seed(1)
data_path = "../../resources/cnn_data/big_data.pkl"
label_path = "../../resources/cnn_data/big_label.pkl"
# Hyper Parameters
epoch = 15       # 训练整批数据多少次,
batch_size = 512  # 批训练大小
learning_rate = 0.0005
classes = 2

class FactDataLoader(Data.Dataset):
    def __init__(self, x_path, y_path, train=None):
        if train:
            # 随机抽取85%的数据作为训练集
            self.x, temp_x, self.y, temp_y = train_test_split(self.get_container(x_path), self.get_container(y_path), test_size=0.15, random_state=1)
            del temp_x, temp_y
        else:
            # 另外15%的数据作为测试集
            temp_x, self.x, temp_y, self.y = train_test_split(self.get_container(x_path), self.get_container(y_path), test_size=0.15, random_state=1)
            del temp_x, temp_y
        self.size = len(self.y)

    def __getitem__(self, index):
        image = self.x[index]
        label = self.y[index]
        x = torch.FloatTensor(image)
        y = torch.LongTensor([label])
        return x, y

    def __len__(self):
        return self.size

    @staticmethod
    def get_container(path):
        file = open(path, 'rb')
        return pickle.load(file)

# 加载训练集
train_data_set = FactDataLoader(data_path, label_path, train=True)
test_data_set = FactDataLoader(data_path, label_path)

# 将dataset放入DataLoader
train_loader = Data.DataLoader(dataset=train_data_set, batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = Data.DataLoader(dataset=test_data_set, batch_size=batch_size, shuffle=True, num_workers=1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer1 = nn.Linear(5 * 7 * 7, 128)
        self.layer2 = nn.Linear(128, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

if __name__ == '__main__':

    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    target_names = ['class 1', 'class 0']

    for i in range(epoch):
        train_loss, train_precision = 0, 0
        model.train()
        all_pred_y = []
        all_read_y = []
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_y = batch_y.view(-1)
            output = model(batch_x)
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred_y = torch.max(output, 1)[1].data.numpy().tolist()
            batch_y = batch_y.data.numpy().tolist()
            all_pred_y += pred_y
            all_read_y += batch_y
        r = classification_report(all_read_y, all_pred_y, target_names=target_names)
        print(r)
        #     pred_y = torch.max(output, 1)[1]
        #     train_precision += (pred_y.data.numpy() == batch_y.data.numpy()).sum()
        #     train_loss += loss.item()
        #     show_str = ('[%%-%ds]' % 30) % (int(30 * (step * 512) / (26922 * 0.85)) * "#")
        #     print('\r%s %d%%' % (show_str, step * 512 * 100 / (26922 * 0.85)), end="")
        # train_precision = train_precision / len(train_data_set)
        # train_loss = train_loss / step
        # print()
        # print('第{}次训练：loss值：{}准确率：{}'.format(i+1, train_loss, train_precision))

        test_loss, test_precision = 0, 0
        all_pred_y = []
        all_read_y = []
        for step, (batch_x, batch_y) in enumerate(test_loader):
            batch_y = batch_y.view(-1)
            output = model(batch_x)
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred_y = torch.max(output, 1)[1].data.numpy().tolist()
            batch_y = batch_y.data.numpy().tolist()
            all_pred_y += pred_y
            all_read_y += batch_y
        r = classification_report(all_read_y, all_pred_y, target_names=target_names)
        print(r)
        #     pred_y = torch.max(output, 1)[1]
        #     test_precision += (pred_y.data.numpy() == batch_y.data.numpy()).sum()
        #     test_loss += loss.item()
        #     show_str = ('[%%-%ds]' % 30) % (int(30 * (step * 512) / (26922 * 0.15)) * "#")
        #     print('\r%s %d%%' % (show_str, step * 512 * 100 / (26922 * 0.15)), end="")
        #
        # test_precision = test_precision / len(test_data_set)
        # test_loss = test_loss / step
        # print()
        # print('第{}次测试：loss值：{}准确率：{}'.format(i + 1, test_loss, test_precision))

