import copy
import math
import pickle
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
    # users = model.wv.most_similar('1064965', topn=20)
    # users = [i[0] for i in users]
    users = list(snd.get_users_by_batch_original_mid({"3338745751776606", "3338812282191870"}))
    users.remove(326454)
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
        self.center_box = []

    def cut(self):
        max_index = np.argmax(self.data, axis=0)
        min_index = np.argmin(self.data, axis=0)
        # 坐标平移
        self.data[:, 0] -= self.data[min_index[0], 0]
        self.data[:, 1] -= self.data[min_index[1], 1]
        x_max = self.data[max_index[0], 0]
        y_max = self.data[max_index[1], 1]
        self.unit_length = math.sqrt((x_max * y_max) / (3 * self.data[:, 0].size))
        # box plots
        self.columns = int(x_max / self.unit_length) + 1
        self.rows = int(y_max / self.unit_length) + 1
        # 以经验来看self.columns和self.rows相差无几，为方便反向螺旋遍历，将矩阵构建为方阵
        self.rows = self.columns = max(self.rows, self.columns)
        self.boxes = [[[]for i in range(self.columns)] for j in range(self.rows)]
        for item in range(len(self.data)):
            column = int(self.data[item][0] / self.unit_length)
            row = int(self.data[item][1] / self.unit_length)
            self.boxes[self.rows - 1 - row][column].append(tuple(self.data[item]) + (self.users[item],))

        self.center_box = [(self.rows - 1) // 2, (self.columns - 1) // 2]

    def spread_from_origin(self):
        mul_node_in_box = True
        while mul_node_in_box:
            mul_node_in_box = False
            # 从原点（左下角），往右、上方向遍历每一个网格，
            for i in range(self.rows - 1, -1, -1):
                for j in range(self.columns):
                    cur_box = self.boxes[i][j]
                    # 按照与远点距离进行排序
                    cur_box.sort(key=lambda x: math.sqrt(x[0] ** 2 + x[1] ** 2))
                    # 当网格中存在大于1个节点时才需要扩散
                    while len(cur_box) > 1:
                        if j != self.columns - 1 and len(self.boxes[i][j + 1]) == 0:
                            self.boxes[i][j + 1].append(cur_box.pop(1))
                        # 上网格存
                        elif i != 0:
                            self.boxes[i - 1][j].append(cur_box.pop(1))
                        # 没有位置扩散了
                        else:
                            mul_node_in_box = True
                            break
            if mul_node_in_box:
                # 增加行
                self.boxes.insert(0, [[] for i in range(self.columns)])
                self.rows += 1

    """
        从中心网格顺时针螺旋顺序，遍历每一个网格，当网格中存在多个节点时需要往外扩散，
        首先确定网格中的中心点与移动点，移动点按照与中心点的相对位置往周围扩散，
        扩散方向有上、右上、右、右下、下、左下、左、左上8个，但扩散时应遵循两个原则：
        1、不往内层的网格扩散
        2、不往同层已存在节点的网格强制扩散
    """
    def spread_from_center(self):
        mul_node_in_box = True
        while mul_node_in_box:
            mul_node_in_box = False
            cur_box_index = self.center_box.copy()
            direction = [1, 0, 0, 0]  # 右下左上（螺旋方向）
            complete = [cur_box_index[1], cur_box_index[0], cur_box_index[1], cur_box_index[0]]  # 右下左上（行列）
            for i in range(self.rows * self.columns):
                cur_box = self.boxes[cur_box_index[0]][cur_box_index[1]]
                around_box = self.get_box_from_around(cur_box_index)
                # 将周围八个网格分三个等级，0:不可扩散方格(内层)、1:不可强制扩散方格(同层)、2:可强制扩散方格(外层)
                around_box_rank = self.get_box_from_around_rank(cur_box_index, direction, complete)
                while len(cur_box) > 1:
                    center_x = self.center_box[0] + .5
                    center_y = self.center_box[1] + .5
                    cur_box.sort(key=lambda x: math.sqrt((x[0] - center_x) ** 2 + (x[1] - center_y) ** 2))
                    radian = math.atan2((cur_box[1][0] - cur_box[0][0]), (cur_box[1][1] - cur_box[0][1]))
                    angle = radian * (180 / math.pi)
                    if direction[0] == 1:
                        self.right(angle, complete, cur_box_index, cur_box, around_box)

                # 螺旋方向 -> 右
                if direction[0] == 1:
                    cur_box_index[1] += 1
                    # 转向 -> 下
                    if cur_box_index[1] == complete[0] + 1:
                        complete[0] = cur_box_index[1]
                        direction[0] = 0
                        direction[1] = 1
                # 螺旋方向 -> 下
                elif direction[1] == 1:
                    cur_box_index[0] += 1
                    # 转向 -> 左
                    if cur_box_index[0] == complete[1] + 1:
                        complete[1] = cur_box_index[0]
                        direction[1] = 0
                        direction[2] = 1
                # 螺旋方向 -> 左
                elif direction[2] == 1:
                    cur_box_index[1] -= 1
                    #  转向 -> 上
                    if cur_box_index[1] == complete[2] - 1:
                        complete[2] = cur_box_index[1]
                        direction[2] = 0
                        direction[3] = 1
                # 螺旋方向 -> 上
                else:
                    cur_box_index[0] -= 1
                    # 转向 -> 右
                    if cur_box_index[0] == complete[3] - 1:
                        complete[3] = cur_box_index[0]
                        direction[3] = 0
                        direction[0] = 1
            if mul_node_in_box:
                # 增加行
                self.boxes.insert(0, [[] for i in range(self.columns)])
                self.rows += 1

    def get_box_from_around(self, cur_box_index):
        up_box = self.boxes[cur_box_index[0] - 1][cur_box_index[1]]
        right_up_box = self.boxes[cur_box_index[0] - 1][cur_box_index[1] + 1]
        right_box = self.boxes[cur_box_index[0]][cur_box_index[1] + 1]
        right_down_box = self.boxes[cur_box_index[0] + 1][cur_box_index[1] + 1]
        down_box = self.boxes[cur_box_index[0] + 1][cur_box_index[1]]
        left_down_box = self.boxes[cur_box_index[0] + 1][cur_box_index[1] - 1]
        left_box = self.boxes[cur_box_index[0]][cur_box_index[1] - 1]
        left_up_box = self.boxes[cur_box_index[0] - 1][cur_box_index[1] - 1]
        return [
            [left_up_box,   up_box,   right_up_box],
            [left_box,      None,     right_box],
            [left_down_box, down_box, right_down_box]
        ]

    def get_box_from_around_rank(self, cur_box_index, direction, complete):
        if cur_box_index == self.center_box:
            return [
                [2, 2, 2],
                [2, None, 2],
                [2, 2, 2]
            ]
        if direction[0] == 1:
            if cur_box_index[1] == complete[2]:
                return [
                    [2, 2, 2],
                    [2, None, 1],
                    [2, 1, 0]
                ]
            elif cur_box_index[1] == complete[2] + 1:
                return [
                    [2, 2, 2],
                    [1, None, 1],
                    [1, 0, 0]
                ]
            elif complete[2] + 1 < cur_box_index[1] < complete[0]:
                return [
                    [2, 2, 2],
                    [1, None, 1],
                    [0, 0, 0]
                ]
            elif cur_box_index[1] == complete[0]:
                return [
                    [2, 2, 2],
                    [1, None, 1],
                    [0, 0, 1]
                ]
            else:
                print("等级划分错误")
        elif direction[1] == 1:
            if cur_box_index[0] == complete[3]:
                return [
                    [2, 2, 2],
                    [1, None, 2],
                    [0, 1, 2]
                ]
            elif cur_box_index[0] == complete[3] + 1:
                return [
                    [1, 1, 2],
                    [0, None, 2],
                    [0, 1, 2]
                ]
            elif complete[3] + 1 < cur_box_index[0] < complete[1]:
                return [
                    [0, 1, 2],
                    [0, None, 2],
                    [0, 1, 2]
                ]
            elif cur_box_index[0] == complete[0]:
                return [
                    [0, 1, 2],
                    [0, None, 2],
                    [1, 1, 2]
                ]
            else:
                print("等级划分错误")
        elif direction[2] == 1:
            if cur_box_index[1] == complete[0]:
                return [
                    [0, 1, 2],
                    [1, None, 2],
                    [2, 2, 2]
                ]
            elif cur_box_index[1] == complete[0] - 1:
                return [
                    [0, 0, 1],
                    [1, None, 1],
                    [2, 2, 2]
                ]
            elif complete[2] < cur_box_index[1] < complete[0] - 1:
                return [
                    [0, 0, 0],
                    [1, None, 1],
                    [2, 2, 2]
                ]
            elif cur_box_index[1] == complete[2]:
                return [
                    [1, 0, 0],
                    [1, None, 1],
                    [2, 2, 2]
                ]
            else:
                print("等级划分错误")
        elif direction[3] == 1:
            if cur_box_index[1] == complete[1]:
                return [
                    [2, 1, 0],
                    [2, None, 1],
                    [2, 2, 2]
                ]
            elif cur_box_index[1] == complete[1] - 1:
                return [
                    [2, 1, 0],
                    [2, None, 0],
                    [2, 1, 1]
                ]
            elif complete[3] < cur_box_index[1] < complete[1] - 1:
                return [
                    [2, 1, 0],
                    [2, None, 0],
                    [2, 1, 0]
                ]
            elif cur_box_index[1] ==complete[3]:
                return [
                    [2, 1, 1],
                    [2, None, 0],
                    [2, 1, 0]
                ]
        else:
            print("方向计算错误")

    def right(self, angle, complete, cur_box_index, cur_box, around_box):

        if cur_box_index[1] == complete[2]:
            around_box_rank = [
                [2, 2, 2],
                [2, None, 1],
                [2, 1, 0]
            ]
        elif cur_box_index[1] == complete[2] + 1:
            around_box_rank = [
                [2, 2, 2],
                [1, None, 1],
                [1, 0, 0]
            ]
        elif cur_box_index[1] == complete[0]:
            around_box_rank = [
                [2, 2, 2],
                [1, None, 1],
                [0, 0, 1]
            ]
        else:
            around_box_rank = [
                [2, 2, 2],
                [1, None, 1],
                [0, 0, 0]
            ]
        # 往上扩散
        if -22.5 < angle <= 22.5:
            if (around_box_rank[0][1] == 2) or (around_box_rank[0][1] == 1 and len(around_box[0][1] == 0)):
                around_box[0][1].append(cur_box.pop(1))
        # 往右上扩散
        elif 22.5 < angle <= 67.5:
            self.boxes[cur_box_index[0] - 1][cur_box_index[1] + 1].append(cur_box.pop(1))
        # 往右扩散
        elif 67.5 < angle <= 112.5:
            self.boxes[cur_box_index[0]][cur_box_index[1] + 1].append(cur_box.pop(1))
        # 往右下扩散
        elif 112.5 < angle <= 157.5:
            self.boxes[cur_box_index[0] + 1][cur_box_index[1] + 1].append(cur_box.pop(1))
        # 往下扩散
        elif 157.5 < angle or angle <= -157.5:
            self.boxes[cur_box_index[0] + 1][cur_box_index[1]].append(cur_box.pop(1))
        # 往左下扩散
        elif -157.5 < angle <= -112.5:
            self.boxes[cur_box_index[0] + 1][cur_box_index[1] - 1].append(cur_box.pop(1))
        # 往左扩散
        elif -112.5 < angle <= -67.5:
            self.boxes[cur_box_index[0]][cur_box_index[1] - 1].append(cur_box.pop(1))
        # 往左上扩散
        elif -67.5 < angle <= -22.5:
            self.boxes[cur_box_index[0] - 1][cur_box_index[1] - 1].append(cur_box.pop(1))
        else:
            print("方向计算错误")

    def down(self, direction, complete, cur_box_index):
        cur_box_index[0] += 1
        # 转向
        if cur_box_index[0] == complete[1] + 1:
            complete[1] = cur_box_index[0]
            direction[1] = 0
            direction[2] = 1

    def left(self, direction, complete, cur_box_index):
        cur_box_index[1] -= 1
        #  转向
        if cur_box_index[1] == complete[2] - 1:
            complete[2] = cur_box_index[1]
            direction[2] = 0
            direction[3] = 1

    def up(self, direction, complete, cur_box_index):
        cur_box_index[0] -= 1
        # 转向
        if cur_box_index[0] == complete[3] - 1:
            complete[3] = cur_box_index[0]
            direction[3] = 0
            direction[0] = 1
if __name__ == '__main__':
    users_l_vec = np.array([
        [0.0, 0.0],
        [0.2, 0.2],
        [0.6, 0.2],
        [0.3, 1.7],
        [1.5, 2.5],
        [2.5, 2.5],
        [3.6, 3.7],
        [3.65, 3.65],
        [5.0, 4.0],
        [5.0, 6.0]
    ])
    users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # users_l_vec, users = get_data(user_vec_path)
    # pickle.dump(users_l_vec, open("../../resources/users_l_vec.pkl", "wb+"))
    # pickle.dump(users, open("../../resources/users.pkl", "wb+"))
    # users_l_vec = pickle.load(open("../../resources/users_l_vec.pkl", "rb+"))
    # users = pickle.load(open("../../resources/users.pkl", "rb+"))

    rm = User2Pixel(users_l_vec, users)
    rm.cut()
    # rm.spread_from_origin()
    rm.spread_from_center()
    # user_pixel = rm.boxes
    # for row in range(len(user_pixel)):
    #     for column in range(len(user_pixel[0])):
    #         if len(user_pixel[row][column]) == 1:
    #             user_pixel[row][column] = user_pixel[row][column][0][2]
    #         else:
    #             user_pixel[row][column] = -1
    #
    # pickle.dump(user_pixel, open("../../resources/user_pixel.pkl", "wb+"))
    # plt.scatter(users_l_vec[:, 0], users_l_vec[:, 1], marker='.', s=40, label='First')
    # plt.show()
    # user_pixel = pickle.load(open("../../resources/user_pixel.pkl", "rb+"))
    # re = []
    # nore = []
    # for i in range(len(user_pixel)):
    #     for j in range(len(user_pixel[0])):
    #         if user_pixel[i][j] != -1:
    #             re.append([i, rm.rows - 1 - j])
    #         else:
    #             nore.append([i, rm.rows - 1 - j])
    # re = np.array(re)
    # nore = np.array(nore)
    # plt.scatter(re[:, 0], re[:, 1], marker='.', s=40, label='First')
    # plt.scatter(nore[:, 0], nore[:, 1], marker='', s=40, label='First', c="#F0FAFF")
    # plt.show()