# matrix = [
#     [20, 21, 22, 23,24],
#     [19,  6,  7,  8,  9],
#     [18,  5,  0,  1, 10],
#     [17,  4,  3,  2, 11],
#     [17, 15, 14, 13,  12]
# ]
#
# center_box = [(len(matrix) - 1) // 2, (len(matrix[0]) - 1) // 2]
# print(center_box)
# rows = len(matrix)
# columns = len(matrix[0])
# cur_i = center_box[0]  # 行
# cur_j = center_box[1]  # 列
# direction = [1, 0, 0, 0]  # 右下左上（方向）
# ans = []
# spread_complete = [cur_j, cur_i, cur_j, cur_i]  # 右下左上（行列）
# for i in range(rows * columns):
#     ans.append(matrix[cur_i][cur_j])
#     # 往右遍历
#     if direction[0] == 1:
#         # 到达右边界
#         cur_j += 1
#         # 转向
#         if cur_j == spread_complete[0] + 1:
#             spread_complete[0] = cur_j
#             direction[0] = 0
#             direction[1] = 1
#     # 往下遍历
#     elif direction[1] == 1:
#         cur_i += 1
#         # 转向
#         if cur_i == spread_complete[1] + 1:
#             spread_complete[1] = cur_i
#             direction[1] = 0
#             direction[2] = 1
#     # 往左遍历
#     elif direction[2] == 1:
#         cur_j -= 1
#         if cur_j == spread_complete[2] - 1:
#             spread_complete[2] = cur_j
#             direction[2] = 0
#             direction[3] = 1
#     # 往上遍历
#     else:
#         cur_i -= 1
#         # 转向
#         if cur_i == spread_complete[3] - 1:
#             spread_complete[3] = cur_i
#             direction[3] = 0
#             direction[0] = 1
#
#
# print(ans)
import copy
import math
import random

import numpy as np


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
        elif cur_box_index[1] == complete[3]:
            return [
                [2, 1, 1],
                [2, None, 0],
                [2, 1, 0]
            ]
    else:
        print("方向计算错误")

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
        [left_up_box, up_box, right_up_box],
        [left_box, None, right_box],
        [left_down_box, down_box, right_down_box]
    ]
if __name__ == "__main__":
    around_box = [
        [1, 2,1,1,1,1,1,1,1,1,1,1],
        [1, 2,1,1,1,1,1,1,1,1,1,1],
        [1, 2,1,1,1,1,1,1,1,1,1,1],
        [1, 2,1,1,1,1,1,1,1,1,1,1],
        [1, 2,1,1,1,1,9,1,1,1,1,1],
        [1, 2,1,1,1,1,1,1,1,1,1,1],
        [1, 2,1,1,1,1,1,1,1,1,1,1],
        [1, 2,1,1,1,1,1,1,1,1,1,-1],
        [1, 2,1,1,1,1,1,1,1,1,1,0]
    ]
    around_box = np.array(around_box)
    # print(around_box[4-4:4+4+1, 6-4:6+4])
    np.maximum(around_box, 0)
    a = [1,2,3]
    b = [1,5]
    a += b
    print(a)