import math
import time

import numpy as np
import matplotlib.pyplot as plt
import dao.social_networks_dao as snd
from sklearn import manifold
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import KeyedVectors

user_vec_path = "../../resources/model/user_vec.model"
# user_vec_path = "../../resources/model/in_user_vec.model"


def get_data(path):
    model = KeyedVectors.load(path)
    users_h_vec = []
    users = model.wv.most_similar('1064965', topn=20)
    users = [i[0] for i in users]
    # users = snd.get_users_by_batch_original_mid({"3338745751776606", "3338812282191870"})
    # users.remove(326454)
    for user_id in users:
        users_h_vec.append(model[str(user_id)])
    # 使用t-sne降维
    users_h_vec = np.array(users_h_vec)
    t_sne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    users_l_vec = t_sne.fit_transform(users_h_vec)
    return users_l_vec, users

class User2Pixel(object):
    def __init__(self, data, users):
        self.related_distance = 2
        self.data = data
        self.users = users
        self.unit_length = 0
        self.boxes = []
        self.rows = 0
        self.columns = 0

    def cut(self):
        max_index = np.argmax(self.data, axis=0)
        min_index = np.argmin(self.data, axis=0)
        # 坐标平移
        self.data[:, 0] -= self.data[min_index[0], 0]
        self.data[:, 1] -= self.data[min_index[1], 1]
        x_max = self.data[max_index[0], 0]
        y_max = self.data[max_index[1], 1]
        self.unit_length = math.sqrt((x_max * y_max) / (2 * self.data[:, 0].size))
        # box plots
        self.columns = int(x_max / self.unit_length) + 1
        self.rows = int(y_max / self.unit_length) + 1
        self.boxes = [[[]for i in range(self.columns)] for j in range(self.rows)]
        for item in range(len(self.data)):
            column = int(self.data[item][0] / self.unit_length)
            row = int(self.data[item][1] / self.unit_length)
            self.boxes[self.rows - 1 - row][column].append(tuple(self.data[item]) + (self.users[item],))
            # self.boxes[self.rows - 1 - row][column].append(item)

        for i in self.boxes:
            print(i)

    def spread(self):
        mul_node_in_box = True
        while mul_node_in_box:
            mul_node_in_box = False
            # 从左下角，往上、右方向扩散
            for i in range(self.rows - 1, -1, -1):
                for j in range(self.columns):
                    # 遍历每一个网格
                    cur_box = self.boxes[i][j]
                    # 当网格中存在大于1个节点时才需要扩散
                    if len(cur_box) > 1:

                        # 寻找中心节点
                        center_node = cur_box[0]
                        remove_node = []
                        for k in range(1, len(cur_box)):
                            cur_node = cur_box[k]
                            if cur_node[0]*cur_node[1] < center_node[0]*center_node[1]:
                                remove_node.append(center_node)
                                center_node = cur_node
                            else:
                                remove_node.append(cur_node)
                        remove_node.sort(key=lambda x: x[0] * x[1])
                        for tmp_node in remove_node:
                            # 往上、右是否可以推送方格，1, 0, -1表示可推送，可强行推送，不可推送
                            upper = 1
                            right = 1
                            # 到达上边界，不可推送
                            if i == 0:
                                upper = -1
                            # 上方格存在节点，可强行推送
                            elif len(self.boxes[i - 1][j]) != 0:
                                upper = 0
                            # 到达右边界，或右方格存在节点，不可推送
                            if j == self.columns - 1 or len(self.boxes[i][j + 1]):
                                right = -1
                            if upper == 1:
                                self.boxes[i - 1][j].append(tmp_node)
                            elif right == 1:
                                self.boxes[i][j + 1].append(tmp_node)
                            elif upper == 0:
                                self.boxes[i - 1][j].append(tmp_node)
                            else:
                                mul_node_in_box = True
                                break
                            self.boxes[i][j].remove(tmp_node)

            if mul_node_in_box:
                # 增加行、列
                self.boxes.insert(0, [[] for i in range(self.columns)])
                for cur_row in self.boxes:
                    cur_row.append([])
                self.rows += 1
                self.columns += 1

        print("==============")
        for i in self.boxes:
            print(i)

if __name__ == '__main__':

    # s1 = time.time()
    # users_l_vec, users = get_data(user_vec_path)
    s2 = time.time()
    users_l_vec = np.array([
        [0.0, 0.0],
        [0.2, 0.2],
        [0.6, 0.2],
        [0.3, 0.8],
        [0.3, 1.7],
        [1.5, 2.5],
        [5.0, 4.0],
        [5.0, 4.0],
        [5.0, 4.0],
        [5.0, 4.0]
    ])
    users = [0,1,2,3,4,5,6,7,8,9]
    rm = User2Pixel(users_l_vec, users)
    rm.cut()
    s3 = time.time()
    print(s3 - s2)
    rm.spread()
    s4 = time.time()
    print(s4 - s3)
    plt.scatter(users_l_vec[:, 0], users_l_vec[:, 1], marker='.', s=40, label='First')
    plt.show()
