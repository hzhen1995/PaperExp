import copy
import math
import pickle
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import dao.social_networks_dao as snd
from sklearn import manifold
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import KeyedVectors


# user_vec_path = "../../resources/model/in_user_vec.model"


def get_data(params1, params2):
    model = KeyedVectors.load(params1)
    users_h_vec = []
    users = list(snd.get_users_by_batch_original_mid(params2))
    print(len(users))
    # users.remove(326454)
    users.remove(837063)
    users.remove(1271625)
    users.remove(554865)
    for i, user_id in enumerate(users):
        users_h_vec.append(model[str(user_id)])
    print("====")
    # 使用t-sne降维
    users_h_vec = np.array(users_h_vec)
    t_sne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    users_l_vec = t_sne.fit_transform(users_h_vec)
    return users_l_vec, users

class User2Pixel(object):
    def __init__(self, data, users):
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
                # 将周围八个网格分三个等级，0:不可扩散方格(内层)、1:不可强制扩散方格(同层)、2:可强制扩散方格(外层)
                while len(cur_box) > 1:
                    if cur_box_index[0] in [0, self.rows - 1] or cur_box_index[1] in [0, self.columns - 1]:
                        mul_node_in_box = True
                        break
                    cur_box.sort(key=lambda x: math.sqrt(
                        (x[0] - self.center_box[0] - .5) ** 2 + (x[1] - self.center_box[1] - .5) ** 2))
                    radian = math.atan2((cur_box[1][0] - cur_box[0][0]), (cur_box[1][1] - cur_box[0][1]))
                    angle = radian * (180 / math.pi)
                    around_box = self.get_box_from_around(cur_box_index)
                    no_node_around_box = []
                    for k in range(len(around_box)):
                        if len(around_box[k]) == 0:
                            no_node_around_box.append(around_box[k])
                    if direction[0] == 1:
                        self.right(angle, cur_box_index, cur_box, no_node_around_box)
                    elif direction[1] == 1:
                        self.down(angle, cur_box_index, cur_box, no_node_around_box)
                    elif direction[2] == 1:
                        self.left(angle, cur_box_index, cur_box, no_node_around_box)
                    else:
                        self.up(angle, cur_box_index, cur_box, no_node_around_box)
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
                # 在最外层增加一层
                self.boxes.insert(0, [[] for i in range(self.columns)])
                self.boxes.append([[] for i in range(self.columns)])
                for row in self.boxes:
                    row.insert(0, [])
                    row.append([])
                self.rows += 2
                self.columns += 2
                self.center_box = [(self.rows - 1) // 2, (self.columns - 1) // 2]

    def get_box_from_around(self, cur_box_index):
        up_box = self.boxes[cur_box_index[0] - 1][cur_box_index[1]]
        right_up_box = self.boxes[cur_box_index[0] - 1][cur_box_index[1] + 1]
        right_box = self.boxes[cur_box_index[0]][cur_box_index[1] + 1]
        right_down_box = self.boxes[cur_box_index[0] + 1][cur_box_index[1] + 1]
        down_box = self.boxes[cur_box_index[0] + 1][cur_box_index[1]]
        left_down_box = self.boxes[cur_box_index[0] + 1][cur_box_index[1] - 1]
        left_box = self.boxes[cur_box_index[0]][cur_box_index[1] - 1]
        left_up_box = self.boxes[cur_box_index[0] - 1][cur_box_index[1] - 1]
        return [left_up_box, up_box, right_up_box, left_box, right_box, left_down_box, down_box, right_down_box]

    def right(self, angle, cur_box_index, cur_box, no_node_around_box):
        if no_node_around_box:
            to_box = random.choice(no_node_around_box)
            to_box.append(cur_box.pop(1))
            return
        if cur_box_index == self.center_box:
            # 往上扩散
            if -22.5 < angle <= 22.5:
                self.boxes[cur_box_index[0] - 1][cur_box_index[1]].append(cur_box.pop(1))
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
            base_angle, dis_angle = 0, 120
            # 往上扩散
            if base_angle - dis_angle * 0.3 < angle <= base_angle + dis_angle * 0.3:
                self.boxes[cur_box_index[0] - 1][cur_box_index[1]].append(cur_box.pop(1))
            # 往右上扩散
            elif base_angle + dis_angle * 0.3 < angle <= base_angle + dis_angle * 1.5:
                self.boxes[cur_box_index[0] - 1][cur_box_index[1] + 1].append(cur_box.pop(1))
            # 往左上扩散
            else:
                self.boxes[cur_box_index[0] - 1][cur_box_index[1] - 1].append(cur_box.pop(1))

    def down(self, angle, cur_box_index, cur_box, no_node_around_box):
        if no_node_around_box:
            to_box = random.choice(no_node_around_box)
            to_box.append(cur_box.pop(1))
            return
        base_angle, dis_angle = 90, 120
        # 往右扩散
        if base_angle - dis_angle * 0.3 < angle <= base_angle + dis_angle * 0.3:
            self.boxes[cur_box_index[0]][cur_box_index[1] + 1].append(cur_box.pop(1))
        # 往右上扩散
        elif base_angle - dis_angle * 1.5 < angle <= base_angle - dis_angle * 0.3:
            self.boxes[cur_box_index[0] - 1][cur_box_index[1] + 1].append(cur_box.pop(1))
        # 往右下扩散
        else:
            self.boxes[cur_box_index[0] + 1][cur_box_index[1] + 1].append(cur_box.pop(1))

    def left(self, angle, cur_box_index, cur_box, no_node_around_box):
        if no_node_around_box:
            to_box = random.choice(no_node_around_box)
            to_box.append(cur_box.pop(1))
            return
        base_angle, dis_angle = 180, 120
        # 往下扩散
        if base_angle - dis_angle * 0.3 < angle or angle < - base_angle + dis_angle * 0.3:
            self.boxes[cur_box_index[0] + 1][cur_box_index[1]].append(cur_box.pop(1))
        # 往右下扩散
        elif base_angle - dis_angle * 1.5 < angle <= base_angle - dis_angle * 0.3:
            self.boxes[cur_box_index[0] + 1][cur_box_index[1] + 1].append(cur_box.pop(1))
        # 往左下扩散
        else:
            self.boxes[cur_box_index[0] + 1][cur_box_index[1] - 1].append(cur_box.pop(1))

    def up(self, angle, cur_box_index, cur_box, no_node_around_box):
        if no_node_around_box:
            to_box = random.choice(no_node_around_box)
            to_box.append(cur_box.pop(1))
            return
        base_angle, dis_angle = -90, 120
        # 往左扩散
        if base_angle - dis_angle * 0.3 < angle < base_angle + dis_angle * 0.3:
            self.boxes[cur_box_index[0]][cur_box_index[1] - 1].append(cur_box.pop(1))
        # 往左上扩散
        elif base_angle + dis_angle * 0.3 < angle <= base_angle + dis_angle * 1.5:
            self.boxes[cur_box_index[0] - 1][cur_box_index[1] - 1].append(cur_box.pop(1))
        # 往左下扩散
        else:
            self.boxes[cur_box_index[0] + 1][cur_box_index[1] - 1].append(cur_box.pop(1))

if __name__ == '__main__':
    user_vec_path = "../../resources/model/user_vec.model"
    original_mid = {"3339187067795316", "3409823347274485", "3486534206012960",
                    "3338460728295803", "3405653819760021", "3486542481846477"}
    users_l_vec_path = "../../resources/users_l_vec.pkl"
    users_path = "../../resources/users.pkl"
    user_pixel_path = "../../resources/user_pixel.pkl"

    # users_l_vec, users = get_data(user_vec_path, original_mid)
    # pickle.dump(users_l_vec, open(users_l_vec_path, "wb"))
    # pickle.dump(users, open(users_path, "wb"))
    # users_l_vec = pickle.load(open(users_l_vec_path, "rb"))
    # users = pickle.load(open(users_path, "rb"))
    plt.figure(figsize=(6, 5))
    # 原始分布
    # list_user_vec = users_l_vec.tolist()
    # for i in list_user_vec:
    #     plt.scatter(i[0], i[1], 20)
    # rm = User2Pixel(users_l_vec, users)
    # rm.cut()
    # rm.spread_from_origin()
    # rm.spread_from_center()
    # user_pixel = rm.boxes
    # for row in range(rm.rows):
    #     for column in range(rm.columns):
    #         if len(user_pixel[row][column]) >= 1:
    #             user_pixel[row][column] = user_pixel[row][column][0][2]
    #         else:
    #             user_pixel[row][column] = -1
    # pickle.dump(user_pixel, open(user_pixel_path, "wb"))

    user_pixel = pickle.load(open(user_pixel_path, "rb"))
    plt.xticks(range(len(user_pixel)))
    plt.yticks(range(len(user_pixel[0])))
    plt.grid(xdata=range(0, len(user_pixel)), ydata=range(0, len(user_pixel[0])))
    for row in range(len(user_pixel)):
        for column in range(len(user_pixel[0])):
            if user_pixel[row][column] != -1:
                plt.scatter(column, len(user_pixel) - 1 - row, 20)
            else:
                pass
    plt.show()
