import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# 创建数据集
exam_X = DataFrame([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
                   2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
exam_Y = DataFrame([10, 22, 13, 43, 20, 22, 33, 50, 62, 48, 55, 75, 62, 73, 81, 76, 64, 82, 90, 93])

X_train, X_test, Y_train, Y_test = train_test_split(exam_X, exam_Y, train_size=.8)

model = LinearRegression()
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
model.fit(X_train, Y_train)

# 训练数据的预测值
y_train_pred = model.predict(X_train)
print(Y_train)
print(Y_train)
print(y_train_pred)
# 绘制最佳拟合线：标签用的是训练数据的预测值y_train_pred
plt.plot(X_train, y_train_pred, color='black', linewidth=3, label="best line")
# 测试数据散点图
plt.scatter(X_train, Y_train, color='blue', label="train data")
plt.scatter(X_test, Y_test, color='red', label="test data")

# 添加图标标签
plt.legend(loc=2)
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()

score = model.score(X_test, Y_test)
print(X_test)
print(Y_test)
print(score)