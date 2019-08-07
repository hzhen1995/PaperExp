import os
import pickle
import random
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import classification_report


torch.manual_seed(1)
train_path = "../../resources/cnn_data/big_data_train.pkl"
test_path = "../../resources/cnn_data/big_data_test.pkl"
# Hyper Parameters
epoch = 14      # 训练整批数据多少次,
batch_size = 512  # 批训练大小
learning_rate = 0.003
classes = 2

class FactDataLoader(Data.Dataset):

    def __init__(self, data_path):
        self.data = self.get_container(data_path)

    def __getitem__(self, index):
        image = self.data[index][0]
        r = []
        for t in range(3):
            temp = [j for i in image[t] for j in i]
            random.shuffle(temp)
            r += temp
        image = r
        label = self.data[index][1]
        x = torch.FloatTensor(image)
        y = torch.LongTensor([label])
        return x, y

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_container(path):
        file = open(path, 'rb')
        return pickle.load(file)

# 加载训练集
train_set = FactDataLoader(train_path)
test_set = FactDataLoader(test_path)

# 将dataset放入DataLoader
train_loader = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = Data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(675, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

if __name__ == '__main__':
    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    loss_ = []
    pred = []
    f1 = []
    for i in range(epoch):
        train_loss, train_precision = 0, 0
        model.train()
        y_true = []
        y_pred = []
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_y = batch_y.view(-1)
            output = model(batch_x)
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred += torch.max(output, 1)[1].data.numpy().tolist()
            y_true += batch_y.data.numpy().tolist()

            train_precision += (torch.max(output, 1)[1].data.numpy() == batch_y.data.numpy()).sum()
            train_loss += loss.item()
            show_str = ('[%%-%ds]' % 30) % (int(30 * (step * batch_size) / len(train_set)) * "#")
            print('\r%s %d%%' % (show_str, step * batch_size * 100 / len(train_set)), end="")
        evaluate = classification_report(y_true, y_pred, target_names=['class 1', 'class 0'])
        print("\n" + evaluate)
        train_precision = train_precision / len(train_set)
        train_loss = train_loss / step
        print('第{}次训练：loss值：{}准确率：{}'.format(i+1, train_loss, train_precision))
        loss_.append(train_loss)
        pred.append(train_precision)
        f1.append(evaluate[-15:-10])
    print(loss_)
    print(pred)
    print(f1)