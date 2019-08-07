import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# 通过read_csv来读取我们的目的数据集
adv_data = pd.read_csv("../../resources/Advertising.csv")
# 清洗不需要的数据
new_adv_data = adv_data.ix[:, 1:]

X_train, X_test, Y_train, Y_test = train_test_split(new_adv_data.ix[:, :3], new_adv_data.sales, train_size=.80)
model = LinearRegression()

model.fit(X_train, Y_train)
a = model.intercept_  # 截距
b = model.coef_  # 回归系数
print("最佳拟合线:截距", a, ",回归系数：", b)
score = model.score(X_test, Y_test)
print(score)
# 对线性回归进行预测
Y_pred = model.predict(X_test)
print(Y_pred)
plt.plot(range(len(Y_pred)), Y_pred, 'b', label="predict")
# 显示图像
plt.savefig("predict.jpg")
plt.show()