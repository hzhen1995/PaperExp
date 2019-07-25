# -*- coding: utf-8 -*-
import random
from functools import reduce
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle


"""
收缩三原则：
1. 在移动中，只能把离自己远的点移动的更靠近，不能把离自己近的点移动的更远
2. 采用曼哈顿距离 
3. 考虑第一条时不予考虑超过n个距离
"""

"""
收缩方法:采用维特比译码思想
从每个关联极大团开始，
外围点
首先一个点向有利方向移动，即没有远离所有点  
如果没有远离任意一个附近的点
    则进入下轮循环，处理下一个点
否则移动被远离的点，使其保持或缩小之前的距离， 如果m次循环不能保持或缩小则该移动失败


"""


class RejectionModel(object):
    """
    二维坐标上的点放入矩阵中， 思想采用排斥的思想
    """

    def __init__(self, data):
        """
        初始化box，每个box的长度
        :param data: numpy 数组 [num_data, 2]
        """
        # box plots
        # print(data)
        self.related_distance = 2
        self.data = data
        max_index = np.argmax(data, axis=0)
        min_index = np.argmin(data, axis=0)
        # 坐标平移
        data[:, 0] -= data[min_index[0], 0]
        data[:, 1] -= data[min_index[1], 1]
        # print(data)
        horizontal_max = data[max_index[0], 0]
        vertical_max = data[max_index[1], 1]
        # print(horizontal_max, horizontal_min, vertical_max, vertical_min)

        # randomly take sample to calculate density
        import random
        import math
        import pandas as pd
        pd_data = pd.DataFrame(data, columns=['x', 'y'])
        pd_data_describe = pd_data.describe()
        print(pd_data_describe)
        horizontal_mean = pd_data_describe.loc['mean'].x
        vertical_mean = pd_data_describe.loc['mean'].y
        density_list = []
        for i in range(5):
            x = random.gauss(horizontal_mean, math.sqrt(horizontal_max))
            y = random.gauss(vertical_mean, math.sqrt(vertical_max))
            temp_range = horizontal_max / 20
            pd_data_index = \
                (pd_data.x > x - temp_range) \
                & (pd_data.x < x + temp_range) \
                & (pd_data.y > y - temp_range) \
                & (pd_data.y < y + temp_range)
            density_list.append(pd_data[pd_data_index].count().x / temp_range ** 2)
        density = sum(density_list) / len(density_list)
        self.unit_length = int(math.sqrt(3 / 1))
        # print(unit_length)

        # box plots
        horizontal_box = range(0, int(horizontal_max) + 1, self.unit_length)
        vertical_box = range(0, int(vertical_max) + 1, self.unit_length)
        self.boxes = [[[] for j in vertical_box] for i in horizontal_box]
        for plot in range(len(data)):
            x = int(data[plot][0] // self.unit_length)
            y = int(data[plot][1] // self.unit_length)
            self.boxes[x][y].append(plot)
        print(self.boxes)

        # rejection
        #        north
        # west            east
        #        south
        self.start_box_index = [int(len(self.boxes) / 2), int(len(self.boxes[0]) / 2)]
        # center_point_axis = np.array([self.unit_length*i for i in [self.start_point[0]+.5, self.start_point[1]+.5]])

    def is_box_on_side(self, box_index):
        """
        判断一个box是否在边上
        :param box_index: box的坐标
        :return:
        """
        if box_index[0] == 0 or box_index[1] == 1:
            return True
        if box_index[0] > len(self.boxes) or box_index[1] > len(self.boxes[0]):
            return True

    def get_surround_box_index(self, box_index):
        """
        根据给定的box坐标返回其周围八个相邻box
        :param box_index:
        :return:
        """
        x_index = box_index[0]
        y_index = box_index[1]
        west_index = [x_index, y_index - 1]
        northwest_index = [x_index - 1, y_index - 1]
        north_index = [x_index - 1, y_index]
        northeast_index = [x_index - 1, y_index + 1]
        east_index = [x_index, y_index + 1]
        southeast_index = [x_index + 1, y_index + 1]
        south_index = [x_index + 1, y_index]
        southwest_index = [x_index + 1, y_index - 1]

        west_index = self.check_box_index_legal(west_index)
        northwest_index = self.check_box_index_legal(northwest_index)
        north_index = self.check_box_index_legal(north_index)
        northeast_index = self.check_box_index_legal(northeast_index)
        east_index = self.check_box_index_legal(east_index)
        southeast_index = self.check_box_index_legal(southeast_index)
        south_index = self.check_box_index_legal(south_index)
        southwest_index = self.check_box_index_legal(southwest_index)
        return west_index, northwest_index, north_index, northeast_index, \
               east_index, southeast_index, south_index, southwest_index

    def get_surround_vertical_box_index(self, box_index):
        """
        根据给定的box坐标返回其周围横向和纵向四个坐标
        :param box_index:
        :return:
        """
        x_index = box_index[0]
        y_index = box_index[1]
        west_index = [x_index, y_index - 1]
        north_index = [x_index - 1, y_index]
        east_index = [x_index, y_index + 1]
        south_index = [x_index + 1, y_index]

        west_index = self.check_box_index_legal(west_index)
        north_index = self.check_box_index_legal(north_index)
        east_index = self.check_box_index_legal(east_index)
        south_index = self.check_box_index_legal(south_index)

        return west_index, north_index, east_index, south_index

    def check_box_index_legal(self, check_point_index):
        """
        检查点的坐标是否合法，
        :param check_point_index: 检查坐标
        :return: 如果合法返回原值，如果不合法返回[-1,-1]
        """
        x_range = len(self.boxes) - 1
        y_range = len(self.boxes[0]) - 1
        if check_point_index[0] < 0 or check_point_index[1] < 0:
            return [-1, -1]
        elif check_point_index[0] > x_range or check_point_index[1] > y_range:
            return [-1, -1]
        else:
            return check_point_index

    def handel_one_box(self, box_index, direction):
        """
        处理一个点
        :param box_index:
        :param direction:
        :return:
        """
        x_index = box_index[0]
        y_index = box_index[1]

        if len(self.boxes[x_index][y_index]) < 2:
            return
        else:
            box_center_axis = np.array([self.unit_length * i for i in [x_index + .5, y_index + .5]])
            direction_absolute_sum = sum([abs(i) for i in direction])
            point_list = self.boxes[x_index][y_index]
            point_list_temp = self.boxes[x_index][y_index].copy()
            point_distance = \
                [euclidean_distance_for_numpy_array(self.data[i], box_center_axis) for i in point_list_temp]

            # 需要处理坐标不属于该box的，待完成
            point_list_temp_min_index = point_distance.index(min(point_distance))
            point_list_temp.pop(point_list_temp_min_index)

            self.boxes[x_index][y_index] = [point_list[point_list_temp_min_index]]

            # 非全方向
            if direction_absolute_sum > 0:

                for moving_point in point_list_temp:
                    if not self.try_put_element_towards_direction(box_index,
                                                                  moving_point,
                                                                  tuple([-i for i in direction])):
                        self.put_element_towards_direction(box_index, moving_point, direction)
            else:
                west_index, northwest_index, north_index, northeast_index, \
                east_index, southeast_index, south_index, southwest_index = self.get_surround_box_index(box_index)

                no_element_box_index_list = []

                def add_empty_box_to_list(box_index_temp):
                    if box_index_temp[0] != -1 and len(self.boxes[box_index_temp[0]][box_index_temp[1]]) == 0:
                        no_element_box_index_list.append(box_index_temp)

                add_empty_box_to_list(west_index)
                add_empty_box_to_list(northwest_index)
                add_empty_box_to_list(north_index)
                add_empty_box_to_list(northeast_index)
                add_empty_box_to_list(east_index)
                add_empty_box_to_list(southeast_index)
                add_empty_box_to_list(south_index)
                add_empty_box_to_list(southwest_index)

                del_index = 0
                for moving_point in point_list_temp:
                    if len(no_element_box_index_list) > 0:
                        empty_element_box_index = no_element_box_index_list.pop()
                        self.boxes[empty_element_box_index[0]][empty_element_box_index[1]].append(moving_point)
                        del_index += 1
                point_list_temp = point_list_temp[del_index:]

                for point_index in point_list_temp:
                    direction = get_direction(self.data[point_index], box_center_axis)
                    self.put_element_towards_direction(box_index, point_index, direction)

    def put_element_towards_direction(self, box_index, element, direction):
        """
        将一个点往某一个方向的box推，强制性的，一定会将这个点推出去
        :param box_index: 点所在的box的坐标
        :param element: 点
        :param direction: 推的方向
        :return:
        """
        boxes = self.get_nearby_boxes_by_direction(box_index, direction)
        if len(boxes) < 1:
            self.boxes[box_index[0]][box_index[1]].append(element)
            return
        boxes_center_axis = [self.get_box_center_axis(box) for box in boxes]
        element_to_boxes_distance = [euclidean_distance_for_numpy_array(np.array(self.data[element]), box_center_axis)
                                     for box_center_axis in boxes_center_axis]
        target_box_index = boxes[element_to_boxes_distance.index(min(element_to_boxes_distance))]
        self.boxes[target_box_index[0]][target_box_index[1]].append(element)

    def try_put_element_towards_direction(self, box_index, element, direction):
        """
        将一个点往某一个方向的box推，尝试性的，返回是否成功推点
        :param box_index: 点所在的box的坐标
        :param element: 点
        :param direction: 方向
        :return: 是否成功推点
        """
        result = False
        boxes = self.get_nearby_boxes_by_direction(box_index, direction)
        for box in boxes:
            if len(self.boxes[box[0]][box[1]]) < 1:
                self.boxes[box[0]][box[1]].append(element)
                result = True
                break
        return result

    def handel_all_box(self):
        """
        处理所有的点， 边缘点除外
        :return:
        """
        self.handel_one_box(self.start_box_index, (0, 0))
        # plt.scatter(self.start_box_index[0], self.start_box_index[1])
        next_batch_boxes_index = self.get_surround_box_index(self.start_box_index)
        # plt.scatter([i[0] for i in next_batch_boxes_index], [i[1] for i in next_batch_boxes_index])
        # print('start box', self.start_box_index)
        # last_next_batch_boxes_temp = set()
        # plt.ion()
        while len(next_batch_boxes_index) > 0:
            next_batch_boxes_index = self.handle_batch_boxes(next_batch_boxes_index)
            # plt.scatter([i[0] for i in next_batch_boxes_index], [i[1] for i in next_batch_boxes_index])
            # plt.pause(0.2)
            # next_batch_boxes_index = next_batch_boxes_index - (next_batch_boxes_index & last_next_batch_boxes_temp)
            # last_next_batch_boxes_temp = next_batch_boxes_index.copy()
        # self.show_boxes()
        # plt.ioff()
        # plt.show()

    def handle_batch_boxes(self, boxes_index):
        """
        处理一批box
        :param boxes_index:
        :return: 下一批需要处理的box
        """
        next_batch_boxes = set()
        for box_index in boxes_index:
            direction = self.get_direction_for_box(box_index)
            self.handel_one_box(box_index, direction)
            possible_boxes = self.get_nearby_boxes_by_direction(box_index, direction)
            for d in possible_boxes:
                next_batch_boxes.add(tuple(d))

        def check_clear(boxes_check_index):
            """
            检查这一批box是否处理完成"
            :param boxes_check_index: box index list
            :return:
            """
            for bi in boxes_check_index:
                if len(self.boxes[bi[0]][bi[1]]) > 1:
                    if not self.is_box_on_side(bi):
                        return True

        flag = check_clear(boxes_index)
        while flag:
            for box_index in boxes_index:
                direction = self.get_direction_for_box(box_index)
                self.handel_one_box(box_index, direction)
            flag = check_clear(boxes_index)

        return next_batch_boxes - set([tuple(i) for i in boxes_index])

    def get_direction_for_box(self, box_index):
        """
        获取一个box的扩散方向
        :param box_index:
        :return:
        """
        return get_direction(np.array(box_index), np.array(np.array(self.start_box_index)))

    def handle_side_boxes(self):
        """
        处理边缘box
        :return:
        """
        row_count = len(self.boxes)
        column_count = len(self.boxes[0])

        def try_put_point_into_surround_boxes(box_index):
            surround_boxes = self.get_surround_box_index(box_index)
            surround_boxes = [b for b in [self.check_box_index_legal(c) for c in surround_boxes]
                              if b[0] != -1 and len(self.boxes[b[0]][b[1]]) < 1]
            for sb in surround_boxes:
                if len(self.boxes[box_index[0]][box_index[1]]) > 1:
                    self.boxes[sb[0]][sb[1]].append(self.boxes[box_index[0]][box_index[1]].pop())

        def get_direction_cliques_and_move(start_box_index, direction):
            clique = []
            while len(self.boxes[start_box_index[0]][start_box_index[1]]) > 0:
                clique.append(tuple(start_box_index))
                start_box_index = [start_box_index[0] + direction[0], start_box_index[1] + direction[1]]
                if self.check_box_index_legal(start_box_index)[0] < 0:
                    return False
            return self.move_boxes_towards_direction(clique, direction)

        # first row boxes
        for i in range(column_count):
            if len(self.boxes[0][i]) > 1:
                try_put_point_into_surround_boxes([0, i])
                while len(self.boxes[0][i]) > 1:
                    if len(self.boxes[0][i]) > 1:
                        if get_direction_cliques_and_move([0, i + 1], (0, 1)):
                            self.boxes[0][i + 1].append(
                                self.boxes[0][i].pop()
                            )
                    if len(self.boxes[0][i]) > 1:
                        if get_direction_cliques_and_move([0, i - 1], (0, -1)):
                            self.boxes[0][i - 1].append(
                                self.boxes[0][i].pop()
                            )
                    if len(self.boxes[0][i]) > 1:
                        if get_direction_cliques_and_move([1, i - 1], (0, -1)):
                            self.boxes[0][i - 1].append(
                                self.boxes[0][i].pop()
                            )
                    if len(self.boxes[0][i]) > 1:
                        if get_direction_cliques_and_move([1, i + 1], (0, 1)):
                            self.boxes[1][i + 1].append(
                                self.boxes[0][i].pop()
                            )

        # last row boxes
        for i in range(column_count):
            if len(self.boxes[row_count - 1][i]) > 1:
                try_put_point_into_surround_boxes([row_count - 1, i])
                while len(self.boxes[row_count - 1][i]) > 1:
                    if len(self.boxes[row_count - 1][i]) > 1:
                        if get_direction_cliques_and_move([row_count - 1, i + 1], (0, 1)):
                            self.boxes[row_count - 1][i + 1].append(
                                self.boxes[row_count - 1][i].pop()
                            )
                    if len(self.boxes[row_count - 1][i]) > 1:
                        if get_direction_cliques_and_move([row_count, i - 1], (0, -1)):
                            self.boxes[row_count - 1][i - 1].append(
                                self.boxes[row_count - 1][i].pop()
                            )
                    if len(self.boxes[row_count - 1][i]) > 1:
                        if get_direction_cliques_and_move([row_count - 2, i - 1], (0, -1)):
                            self.boxes[row_count - 2][i - 1].append(
                                self.boxes[row_count - 2][i].pop()
                            )
                    if len(self.boxes[row_count - 1][i]) > 1:
                        if get_direction_cliques_and_move([row_count - 2, i + 1], (0, 1)):
                            self.boxes[row_count - 2][i + 1].append(
                                self.boxes[row_count - 2][i].pop()
                            )

        # first column
        for i in range(row_count):
            if len(self.boxes[i][0]) > 1:
                try_put_point_into_surround_boxes([i, 0])
                while len(self.boxes[i][0]) > 1:
                    if len(self.boxes[i][0]) > 1:
                        if get_direction_cliques_and_move([i + 1, 0], (1, 0)):
                            self.boxes[i + 1][0].append(
                                self.boxes[i][0].pop()
                            )
                    if len(self.boxes[i][0]) > 1:
                        if get_direction_cliques_and_move([i - 1, 0], (-1, 0)):
                            self.boxes[i - 1][0].append(
                                self.boxes[i][0].pop()
                            )
                    if len(self.boxes[i][0]) > 1:
                        if get_direction_cliques_and_move([i + 1, 1], (1, 0)):
                            self.boxes[i + 1][1].append(
                                self.boxes[i][0].pop()
                            )
                    if len(self.boxes[i][0]) > 1:
                        if get_direction_cliques_and_move([i - 1, 1], (-1, 0)):
                            self.boxes[i - 1][1].append(
                                self.boxes[i][0].pop()
                            )

        # last column
        for i in range(row_count):
            if len(self.boxes[i][column_count - 1]) > 1:
                try_put_point_into_surround_boxes([i, column_count - 1])
                while len(self.boxes[i][column_count - 1]) > 1:
                    if len(self.boxes[i][column_count - 1]) > 1:
                        if get_direction_cliques_and_move([i + 1, column_count - 1], (1, 0)):
                            self.boxes[i + 1][column_count - 1].append(
                                self.boxes[i][column_count - 1].pop()
                            )
                    if len(self.boxes[i][column_count - 1]) > 1:
                        if get_direction_cliques_and_move([i - 1, column_count - 1], (-1, 0)):
                            self.boxes[i - 1][column_count - 1].append(
                                self.boxes[i][column_count - 1].pop()
                            )
                    if len(self.boxes[i][column_count - 1]) > 1:
                        if get_direction_cliques_and_move([i + 1, column_count - 2], (1, 0)):
                            self.boxes[i + 1][column_count - 2].append(
                                self.boxes[i][column_count - 1].pop()
                            )
                    if len(self.boxes[i][column_count - 1]) > 1:
                        if get_direction_cliques_and_move([i - 1, column_count - 2], (-1, 0)):
                            self.boxes[i - 1][column_count - 2].append(
                                self.boxes[i][column_count - 1].pop()
                            )

    def get_nearby_boxes_by_direction(self, box_index, direction):
        possible_direction = [[direction[0], direction[1]]]
        direction_sum = sum([abs(i) for i in direction])
        if direction_sum > 1:
            possible_direction.append([direction[0], 0])
            possible_direction.append([0, direction[1]])
        else:
            if direction[0] == 0:
                possible_direction.append([1, direction[1]])
                possible_direction.append([-1, direction[1]])
            else:
                possible_direction.append([direction[0], 1])
                possible_direction.append([direction[0], -1])
        possible_boxes = [[box_index[0] + i[0], box_index[1] + i[1]] for i in possible_direction]
        possible_boxes = [self.check_box_index_legal(box) for box in possible_boxes]
        possible_boxes = [[int(box[0]), int(box[1])] for box in possible_boxes]
        return [box for box in possible_boxes if box[0] != -1]

    def get_box_center_axis(self, box_index):
        return np.array([self.unit_length * i for i in [box_index[0] + .5, box_index[1] + .5]])

    def get_related_boxes_in_manhanttan_distance_specified_boxes(self, origin_box, boxes, ):
        """
        获取在指定曼哈顿距离之内的，有值的box
        :param origin_box:
        :param boxes: 指定的方格信息
        :return: box的坐标列表
        """
        return [(x, y) for x in [origin_box[0] + i for i in range(-self.related_distance, self.related_distance + 1)]
                for y in [origin_box[1] + i for i in range(-self.related_distance, self.related_distance + 1)]
                if manhattan_distance(origin_box, (x, y)) <= self.related_distance
                and self.check_box_index_legal((x, y))[0] != -1
                and len(boxes[x][y]) > 0 and tuple(origin_box) != (x, y)]

    def get_related_boxes_in_manhanttan_distance_specified_boxes_in_clique(self, origin_box, boxes, clique_info):
        """
        获取在指定曼哈顿距离之内的，有值的box, 并且在clique中
        :param origin_box:
        :param boxes: 指定的方格信息
        :param clique_info: 团
        :return: box的坐标列表
        """
        return [(x, y) for x in [origin_box[0] + i for i in range(-self.related_distance, self.related_distance + 1)]
                for y in [origin_box[1] + i for i in range(-self.related_distance, self.related_distance + 1)]
                if manhattan_distance(origin_box, (x, y)) <= self.related_distance
                and self.check_box_index_legal((x, y))[0] != -1
                and len(boxes[x][y]) > 0 and tuple(origin_box) != (x, y)
                and (x, y) in clique_info]

    def get_related_boxes_in_manhanttan_distance(self, origin_box):
        return self.get_related_boxes_in_manhanttan_distance_specified_boxes(origin_box, self.boxes)

    def naive_shrink(self, considerate_step=3):
        """
        朴素收缩算法
        1 寻找极大关联团
        2 对每个极大团的点进行收缩
            a) 从外围点开始
            b) 维特比译码思想，考虑计算移动m步后，所有情况中最好的移动方法
            c) 移动评价: 首先看所有移动减少的点与点之间的距离， 其次看这个移动带来的关联极大团的整体紧凑度
        :return:
        """

        def get_one_step_move_info(one_step_move, boxes_info, clique_info):
            """
            计算出这一步的移动的相关信息：移动增益、远离的点，以及移动后的方格信息
            :param one_step_move:
            :param boxes_info:
            :param clique_info:
            :return: move_gain, must_move, boxes_info
            """
            box_temp = deepcopy(boxes_info)
            origin_box = one_step_move[0]
            destination_box = one_step_move[1]
            _related_valuable_boxes = \
                self.get_related_boxes_in_manhanttan_distance_specified_boxes_in_clique(origin_box,
                                                                                        boxes_info,
                                                                                        clique_info)
            _origin_distance = [manhattan_distance(origin_box, rvb) for rvb in _related_valuable_boxes]
            _destiny_distance = [manhattan_distance(destination_box, rvb) for rvb in _related_valuable_boxes]
            _move_gain = [x[0] - x[1] for x in zip(_origin_distance, _destiny_distance)]
            _away_boxes_indexes = [i for i, v in enumerate(_move_gain) if v < 0]
            _must_move = dict([(_related_valuable_boxes[i], [(destination_box, _origin_distance[i])])
                               for i in _away_boxes_indexes])
            box_temp[destination_box[0]][destination_box[1]].append(box_temp[origin_box[0]][origin_box[1]].pop())
            return sum(_move_gain), _must_move, box_temp

        def get_must_move_possible(must_move_info, box_temp):
            """
            根据必须移动信息获取可以落脚的方格位置
            :param must_move_info: [origin, [(destination, distance),,,]]
            :param box_temp: 移动后方格信息
            :return: 可能的目标位置
            """
            origin = must_move_info[0]
            possible_target_move_o = [(x, y) for x in
                                      [origin[0] + i for i in range(-self.related_distance, self.related_distance + 1)]
                                      for y in
                                      [origin[1] + i for i in range(-self.related_distance, self.related_distance + 1)]
                                      if manhattan_distance(origin, (x, y)) <= self.related_distance
                                      and self.check_box_index_legal((x, y))[0] != -1
                                      and len(box_temp[x][y]) < 1
                                      and origin != (x, y)]
            possible_target_move_d_list = []
            for mm in must_move_info[1]:
                destination = mm[0]
                distance_od = mm[1]
                possible_target_move_d_list.append(
                    set([(x, y) for x in [destination[0] + i for i in
                                          range(-self.related_distance, self.related_distance + 1)]
                         for y in [destination[1] + i for i in
                                   range(-self.related_distance, self.related_distance + 1)]
                         if manhattan_distance(destination, (x, y)) <= distance_od
                         and self.check_box_index_legal((x, y))[0] != -1
                         and len(box_temp[x][y]) < 1]))
            possible_target_move_d = list(reduce(lambda x, y: x & y, possible_target_move_d_list))

            return [x for x in possible_target_move_d if x in possible_target_move_o]

        def follow_move(route_info, iteration, clique_info):

            def hand_inner_move(inner_temp_route_list, inner_clique_info):
                """
                根据temp_route进行迭代的循环完成 inner_must_move 移动， 直到inner_must_move为空, 或者inner_must_move不能完成
                :param inner_clique_info:
                :param inner_temp_route_list:
                :return:
                """
                return_route_list = []
                inner_route_clear_flag = True
                for inner_temp_route in inner_temp_route_list:
                    if len(inner_temp_route['inner_must_move']) == 0:
                        return_route_list.append(inner_temp_route)
                        continue
                    else:
                        inner_route_clear_flag = False
                        must_move_first = list(inner_temp_route['inner_must_move'].items())[0]
                        possible_target_boxes = get_must_move_possible(must_move_first,
                                                                       inner_temp_route['temp_box'])
                        if len(possible_target_boxes) == 0:
                            continue
                        for ptb in possible_target_boxes:
                            inner_temp_gain, inner_temp_must_move, inner_temp_boxes = \
                                get_one_step_move_info((must_move_first[0], ptb),
                                                       inner_temp_route['temp_box'],
                                                       inner_clique_info)
                            next_inner_must_move = deepcopy(inner_temp_route['inner_must_move'])
                            next_inner_must_move.pop(must_move_first[0])
                            next_outer_must_move = deepcopy(inner_temp_route['outer_must_move'])

                            for tmm in inner_temp_must_move.items():
                                if tmm[0] in next_inner_must_move:
                                    next_inner_must_move[tmm[0]] += tmm[1]
                                else:
                                    if tmm[0] in next_outer_must_move:
                                        next_outer_must_move[tmm[0]] += tmm[1]
                                    else:
                                        next_outer_must_move[tmm[0]] = tmm[1]
                            return_route_list.append(
                                {
                                    'temp_gain': inner_temp_route['temp_gain'] + inner_temp_gain,
                                    'temp_move': deepcopy(inner_temp_route['temp_move']),
                                    'outer_must_move': next_outer_must_move,
                                    'inner_must_move': next_inner_must_move,
                                    'temp_box': inner_temp_boxes
                                }
                            )
                            return_route_list[-1]['temp_move'].append((must_move_first[0], ptb))

                if inner_route_clear_flag:
                    return return_route_list
                else:
                    return hand_inner_move(return_route_list, clique_info)

            return_route = []
            for single_route in route_info:
                if len(single_route['must_move']) == 0:
                    return_route.append(single_route)
                    continue

                # 根据每条移动信息完成移动
                box_temp = deepcopy(self.boxes)
                for mr in single_route['move']:
                    origin = mr[0]
                    destination = mr[1]
                    box_temp[destination[0]][destination[1]].append(box_temp[origin[0]][origin[1]].pop())

                # 处理must move

                # 首先构建移动 temp_route 变量
                must_move = single_route['must_move']
                must_move_origin_first = list(must_move.items())[0]
                mm_possible_target_box = get_must_move_possible(must_move_origin_first, box_temp)
                temp_route = []
                for mmptb in mm_possible_target_box:
                    temp_gain, temp_must_move, temp_boxes = \
                        get_one_step_move_info((must_move_origin_first[0], mmptb), box_temp, clique)
                    inner_must_move = deepcopy(single_route['must_move'])
                    inner_must_move.pop(must_move_origin_first[0])
                    outer_must_move = dict()
                    for tmm in temp_must_move.items():
                        if tmm[0] in inner_must_move:
                            inner_must_move[tmm[0]] += tmm[1]
                        else:
                            if tmm[0] in outer_must_move:
                                outer_must_move[tmm[0]] += tmm[1]
                            else:
                                outer_must_move[tmm[0]] = tmm[1]
                    temp_route.append({'temp_gain': temp_gain,
                                       'temp_move': [(must_move_origin_first[0], mmptb)],
                                       'outer_must_move': outer_must_move,
                                       'inner_must_move': inner_must_move,
                                       'temp_box': temp_boxes})

                temp_route = hand_inner_move(temp_route, clique_info)

                for tr in temp_route:
                    temp_single_route = dict()
                    temp_single_route['gain'] = single_route['gain'] + tr['temp_gain']
                    temp_single_route['move'] = single_route['move'] + tr['temp_move']
                    temp_single_route['must_move'] = tr['outer_must_move']
                    return_route.append(temp_single_route)

            if iteration > considerate_step:
                return list(filter(lambda x: len(x['must_move']) == 0, return_route))
            else:
                return follow_move(return_route, iteration + 1, clique_info)

        cliques = self.find_max_cliques_abstract(self.find_max_related_clique, drop_center_clique=False)
        cliques = [list(x) for x in cliques]
        for clique_index in range(len(cliques)):
            clique = cliques[clique_index]
            # print('-' * 100)
            # print(clique)
            if len(clique) < 2:
                continue
            clique_array = np.array(clique)
            distance = np.sum(np.fabs(clique_array - clique_array.mean(axis=0)), axis=1)
            distance_index = np.argsort(-distance)

            for di in distance_index:
                print(di, clique[di])
                while True:
                    previous_compactness = sum(manhattan_distance(x, y) for x in clique for y in clique)
                    related_valuable_boxes = self.get_related_boxes_in_manhanttan_distance(clique[di])

                    # 寻找第一步移动，可以去的目标方格
                    first_possible_target_move = \
                        [(x, y)
                         for x in
                         [clique[di][0] + i for i in range(-self.related_distance, self.related_distance + 1)]
                         for y in
                         [clique[di][1] + i for i in range(-self.related_distance, self.related_distance + 1)]
                         if manhattan_distance(clique[di], (x, y)) <= self.related_distance
                         and self.check_box_index_legal((x, y))[0] != -1
                         and len(self.boxes[x][y]) < 1]

                    # 排除移动增益均为负的目标方格
                    previous_distance = [manhattan_distance(clique[di], rvb) for rvb in related_valuable_boxes]
                    first_possible_target_move_save_index = []
                    for index in range(len(first_possible_target_move)):
                        destiny_distance = [manhattan_distance(first_possible_target_move[index], rvb)
                                            for rvb in related_valuable_boxes]
                        difference_od = [x[0] < x[1] for x in zip(previous_distance, destiny_distance)]
                        # save move if all difference_od >= 0
                        if not reduce(lambda x, y: x and y, difference_od):
                            first_possible_target_move_save_index.append(index)
                    first_possible_target_move = [first_possible_target_move[i]
                                                  for i in first_possible_target_move_save_index]

                    # 根据第一步移动信息，进行移动，完成路径信息的初始化
                    total_route = []
                    for ftpm in first_possible_target_move:
                        first_move_gain, first_move_must_move = \
                            get_one_step_move_info((clique[di], ftpm), self.boxes, clique)[:2]
                        total_route.append({'gain': first_move_gain,
                                            'move': [(clique[di], ftpm)],
                                            'must_move': first_move_must_move})
                    route = follow_move(total_route, 1, clique)
                    # route = list(filter(lambda x: x['gain'] > 0, route))
                    if len(route) == 0:
                        break

                    # 筛选可能的路线，首先看gain，然后比较compactness
                    sorted_route = sorted(route, key=lambda x: x['gain'], reverse=True)
                    sorted_route = list(filter(lambda x: x['gain'] == sorted_route[0]['gain'], sorted_route))

                    # if len(sorted_route):
                    #     final_route = sorted_route[0]
                    # else:
                    # 计算compactness
                    for route in sorted_route:
                        clique_temp = deepcopy(clique)
                        for move in route['move']:
                            clique_temp[clique_temp.index(move[0])] = move[1]
                        route['compactness'] = sum([manhattan_distance(x, y) for x in clique_temp for y in clique_temp])
                    final_route = max(sorted_route, key=lambda x: x['compactness'])

                    if final_route['gain'] < 1:
                        if previous_compactness <= final_route['compactness']:
                            break

                    # 根据final_route 移动self.boxes, 更新clique
                    for sm in final_route['move']:
                        # 移动 self.boxes
                        self.boxes[sm[1][0]][sm[1][1]].append(self.boxes[sm[0][0]][sm[0][1]].pop())

                        # 更新clique
                        clique[clique.index(sm[0])] = sm[1]

                    # plt.figure(figsize=(3, 3))
                    # plt.scatter([x[0] for x in clique], [y[1] for y in clique])
                    # plt.show()
                    # print(clique)

    def clique_shrink_after_naive_shrink(self):
        """
        将每个团向中心收缩
        :return:
        """

        # def get_inner_to_outer_space(_clique_center, _box_center):
        #     """
        #     将团由里向外放，获得较大收缩程度
        #     :return:
        #     """
        #     if 0 == (_clique_center[0] - _box_center[0]):
        #
        #
        #     pass
        cliques = self.find_max_cliques_abstract(self.find_max_related_clique)
        box_center = [len(self.boxes) // 2, len(self.boxes[0]) // 2]
        cliques_array_list = [np.sum(np.abs(np.array(x).mean(axis=0) - np.array(box_center))) for x in cliques]
        clique_order = np.argsort(np.array(cliques_array_list))
        for order in clique_order:
            clique = cliques[order]
            clique_center = np.mean(np.array(clique), axis=0).tolist()
            direction = [-int(i) for i in self.get_direction_for_box(clique_center)]
            can_move, true_direction = self.move_boxes_towards_crude_direction(clique, direction)
            while can_move:
                clique = [[i[0] + true_direction[0], i[1] + true_direction[1]] for i in clique]
                direction = [-int(i) for i in self.get_direction_for_box(clique_center)]
                can_move, true_direction = self.move_boxes_towards_crude_direction(clique, direction)

        def is_boxes_efficient(boxes_index):
            # print(boxes_index)
            for bi in boxes_index:
                if len(self.boxes[bi[0]][bi[1]]) > 0:
                    return True
            return False

        while not is_boxes_efficient([[0, i] for i in range(len(self.boxes[0]))]):
            self.boxes = self.boxes[1:]

        while not is_boxes_efficient([[i, 0] for i in range(len(self.boxes))]):
            for row_index in range(len(self.boxes)):
                self.boxes[row_index] = self.boxes[row_index][1:]

        while not is_boxes_efficient([[len(self.boxes) - 1, i] for i in range(len(self.boxes[0]))]):
            self.boxes = self.boxes[:-1]

        while not is_boxes_efficient([[i, len(self.boxes[0]) - 1] for i in range(len(self.boxes))]):
            for row_index in range(len(self.boxes)):
                self.boxes[row_index] = self.boxes[row_index][:-1]

    def shrink(self):
        """
        扩散后的收缩算法,寻找极大团，将极大团向中间靠拢
        :return:
        """
        cliques = self.find_max_cliques_abstract(self.find_max_clique)
        while len(cliques) > 0:
            clique = cliques.pop()
            clique_size = len(clique)
            clique_center = [int(sum([i[0] for i in clique]) / clique_size),
                             int(sum([i[1] for i in clique]) / clique_size)]
            direction = [-int(i) for i in self.get_direction_for_box(clique_center)]
            self.move_boxes_towards_direction(clique, direction)
            cliques = self.find_max_cliques_abstract(self.find_max_clique)

        def is_boxes_efficient(boxes_index):
            # print(boxes_index)
            for bi in boxes_index:
                if len(self.boxes[bi[0]][bi[1]]) > 0:
                    return True
            return False

        while not is_boxes_efficient([[0, i] for i in range(len(self.boxes[0]))]):
            self.boxes = self.boxes[1:]

        while not is_boxes_efficient([[i, 0] for i in range(len(self.boxes))]):
            for row_index in range(len(self.boxes)):
                self.boxes[row_index] = self.boxes[row_index][1:]

        while not is_boxes_efficient([[len(self.boxes) - 1, i] for i in range(len(self.boxes[0]))]):
            self.boxes = self.boxes[:-1]

        while not is_boxes_efficient([[i, len(self.boxes[0]) - 1] for i in range(len(self.boxes))]):
            for row_index in range(len(self.boxes)):
                self.boxes[row_index] = self.boxes[row_index][:-1]

    def find_max_cliques_abstract(self, clique_find_method, drop_center_clique=True):
        """
        寻找所有极大团
        :return:
        """
        cliques = set()
        efficient_boxes_index = []
        for row in range(len(self.boxes)):
            for column in range(len(self.boxes[0])):
                efficient_boxes_index.append((row, column))
        efficient_boxes_index = set([tuple(i) for i in efficient_boxes_index if len(self.boxes[i[0]][i[1]]) > 0])
        while len(efficient_boxes_index) > 0:
            clique = set(clique_find_method(efficient_boxes_index.pop()))
            cliques.add(tuple(clique))
            efficient_boxes_index -= set(clique)

        # 删除包含中心点的团
        if not drop_center_clique:
            return list(cliques)

        # clique_remove = None
        # for i in cliques:
        #     if tuple(self.start_box_index) in i:
        #         clique_remove = i
        # if clique_remove is not None:
        #     cliques.remove(clique_remove)

        # return list(cliques)
        return list(filter(lambda x: tuple(self.start_box_index) not in x, list(cliques)))

    def find_max_clique(self, box_index):
        """
        寻找非空位置的极大团
        :param box_index: 起始点坐标
        :return:
        """
        max_clique = {tuple(box_index)}
        if len(self.boxes[box_index[0]][box_index[1]]) < 1:
            return {}

        def get_surround_efficient_boxes_by_boxes(boxes_index):
            return_boxes = []
            for bi in boxes_index:
                return_boxes_temp = [tuple(i) for i in self.get_surround_box_index(bi)
                                     if self.check_box_index_legal(i)[0] != -1]
                return_boxes += [i for i in return_boxes_temp if len(self.boxes[i[0]][i[1]]) > 0]
            return set(return_boxes)

        surround_boxes = get_surround_efficient_boxes_by_boxes(max_clique)
        max_clique |= surround_boxes
        while len(surround_boxes) > 0:
            surround_boxes = get_surround_efficient_boxes_by_boxes(surround_boxes) - max_clique
            max_clique |= surround_boxes

        max_clique = list(max_clique)
        max_clique.sort()
        return max_clique

    def find_vertical_max_clique(self, box_index):
        """
        寻找非空位置的垂直极大团
        :param box_index:
        :return:
        """
        max_vertical_clique = {tuple(box_index)}
        if len(self.boxes[box_index[0]][box_index[1]]) < 1:
            return {}

        def get_vertical_surround_efficient_boxes_by_boxes(boxes_index):
            return_boxes = []
            for bi in boxes_index:
                return_boxes_temp = [tuple(i) for i in self.get_surround_vertical_box_index(bi)
                                     if self.check_box_index_legal(i)[0] != -1]
                return_boxes += [i for i in return_boxes_temp if len(self.boxes[i[0]][i[1]]) > 0]
            return set(return_boxes)

        vertical_surround_boxes = get_vertical_surround_efficient_boxes_by_boxes(max_vertical_clique)
        max_vertical_clique |= vertical_surround_boxes
        while len(vertical_surround_boxes) > 0:
            vertical_surround_boxes = get_vertical_surround_efficient_boxes_by_boxes(vertical_surround_boxes) \
                                      - max_vertical_clique
            max_vertical_clique |= vertical_surround_boxes

        max_vertical_clique = list(max_vertical_clique)
        max_vertical_clique.sort()
        return max_vertical_clique

    def find_max_related_clique(self, box_index, n=2):
        """
        寻找相隔距离小于或等于n的极大团
        :param box_index:
        :param n
        :return:
        """
        max_vertical_clique = {tuple(box_index)}
        if len(self.boxes[box_index[0]][box_index[1]]) < 1:
            return {}

        def get_related_surround_efficient_boxes_by_box(start_box_index, m):
            return [(x, y) for x in [start_box_index[0] + i for i in range(-m, m + 1)]
                    for y in [start_box_index[1] + i for i in range(-m, m + 1)]
                    if manhattan_distance(start_box_index, (x, y)) <= m and self.check_box_index_legal((x, y))[0] != -1]

        def get_related_surround_efficient_boxes_by_boxes(boxes_index, m):
            return_boxes = []
            for bi in boxes_index:
                return_boxes_temp = [tuple(i) for i in get_related_surround_efficient_boxes_by_box(bi, m)]
                return_boxes += [i for i in return_boxes_temp if len(self.boxes[i[0]][i[1]]) > 0]
            return set(return_boxes)

        vertical_surround_boxes = get_related_surround_efficient_boxes_by_boxes(max_vertical_clique, n)
        max_vertical_clique |= vertical_surround_boxes
        while len(vertical_surround_boxes) > 0:
            vertical_surround_boxes = get_related_surround_efficient_boxes_by_boxes(vertical_surround_boxes, n) \
                                      - max_vertical_clique
            max_vertical_clique |= vertical_surround_boxes

        max_vertical_clique = list(max_vertical_clique)
        max_vertical_clique.sort()
        return max_vertical_clique

    def move_boxes_towards_direction(self, boxes_index, towards):
        boxes_index = [tuple(i) for i in boxes_index]
        new_boxes_index = [[i[0] + towards[0], i[1] + towards[1]] for i in boxes_index]
        for nb in new_boxes_index:
            if self.check_box_index_legal(nb)[0] == -1:
                return False
            if len(self.boxes[nb[0]][nb[1]]) > 0:
                if not (nb[0], nb[1]) in boxes_index:
                    return False
        # boxes_data = dict()
        # for box_index in boxes_index:
        #     boxes_data[tuple(box_index)] = self.boxes[box_index[0]][box_index[1]].copy()
        #     self.boxes[box_index[0]][box_index[1]] = []
        #
        # for box_index in boxes_data.keys():
        #     self.boxes[box_index[0] + towards[0]][box_index[1] + towards[1]] = boxes_data[box_index]
        for index in range(len(boxes_index)):
            self.boxes[new_boxes_index[index][0]][new_boxes_index[index][1]].append(
                self.boxes[boxes_index[index][0]][boxes_index[index][1]].pop(0)
            )
        return True

    def move_boxes_towards_crude_direction(self, boxes_index, direction):
        direction_list = []
        if direction == [0, 0]:
            return False, (0, 0)
        if sum([abs(i) for i in direction]) == 1:
            direction_list.append(direction)
            direction_copy = direction.copy()
            direction_copy[direction_copy.index(0)] = -1
            direction_list.append(direction_copy)
            direction_copy = direction.copy()
            direction_copy[direction_copy.index(0)] = 1
            direction_list.append(direction_copy)
        else:
            direction_list.append(direction)
            direction_list.append([direction[0], 0])
            direction_list.append([0, direction[1]])

        for direction in direction_list:
            if self.move_boxes_towards_direction(boxes_index, direction):
                return True, direction
        return False, (0, 0)

    def show_boxes_with_plt(self, name='boxes.eps', title='rejection result', figsize=(6, 6)):
        """
        plt 画出最终的matrix
        :return:
        """
        f = plt.figure(figsize=figsize)
        for x in range(len(self.boxes)):
            for y in range(len(self.boxes[0])):
                if len(self.boxes[x][y]) > 0:
                    # xs.append(x)
                    # ys.append(y)
                    plt.scatter(x, y, 60)
                    # plt.annotate(s=str(self.boxes[x][y][0]),
                    #              xy=(x, y))
        plt.title(title)
        plt.xticks(range(len(self.boxes)))
        plt.yticks(range(len(self.boxes[0])))
        plt.grid(xdata=range(0, len(self.boxes)),
                 ydata=range(0, len(self.boxes[1])))
        # figure.savefig('rejection.png')
        plt.savefig(name, dpi=400)
        plt.show()

    def plot_cliques(self, name='cliques.eps', figsize=(6, 5)):
        plt.figure(figsize=figsize)
        cliques = self.find_max_cliques_abstract(self.find_max_related_clique, drop_center_clique=False)
        for clique in cliques:
            plt.scatter([c[0] for c in clique], [c[1] for c in clique])
        plt.savefig(name, dpi=600)
        plt.show()

    def show_raw_plot(self, name='row plot.eps', figsize=(6, 5)):
        plt.figure(figsize=figsize)
        for p in range(len(self.data)):
            plt.scatter(self.data[p][0], self.data[p][1], 60)
            # plt.annotate(s=str(p),
            #              xy=(self.data[p][0], self.data[p][1]))
        plt.savefig(name, dpi=600)
        plt.show()

    def get_rejection_result(self):
        """
        得到passport2matrix的结果
        :return: x, y, data
        """
        result = list(range(len(self.data)))
        for i in range(len(self.boxes)):
            for j in range(len(self.boxes[i])):
                if len(self.boxes[i][j]) > 0:
                    result[self.boxes[i][j][0]] = (i, j)
        return len(self.boxes), len(self.boxes[0]), result

    def final_shrink(self):
        """
        处理最终的收缩，
        :return:
        """

        def check_points_movable(boxes_index, directions):
            """
            检查这些点能否向这些方向移动一格
            :param boxes_index: box 的坐标
            :param directions: 移动方向， 垂直坐标放第一
            :return: 检查结果
            """
            check_result = True
            moved_boxes_temp = []
            move_plan = []
            for bi in boxes_index:
                single_box_flag = False
                for direct in directions:
                    target_bi = (bi[0] + direct[0], bi[1] + direct[1])
                    if (target_bi not in moved_boxes_temp) and (len(self.boxes[target_bi[0]][target_bi[1]]) < 1):
                        moved_boxes_temp.append(target_bi)
                        single_box_flag = True
                        move_plan.append((bi, target_bi))
                        break
                if not single_box_flag:
                    check_result = False
                    break

            if check_result:
                for mp in move_plan:
                    origin = mp[0]
                    dead = mp[1]
                    self.boxes[dead[0]][dead[1]].append(self.boxes[origin[0]][origin[1]].pop())
            return check_result

        # 处理第一行
        temp_flag = True
        while temp_flag:
            check_boxes_index = []
            for i in range(len(self.boxes[0])):
                if len(self.boxes[0][i]) > 0:
                    check_boxes_index.append((0, i))
            temp_flag = check_points_movable(check_boxes_index, ((1, 0), (1, -1), (1, 1)))
            if temp_flag:
                self.boxes = self.boxes[1:]

        # 处理最后一行
        temp_flag = True
        while temp_flag:
            check_boxes_index = []
            last_row_index = len(self.boxes) - 1
            for i in range(len(self.boxes[last_row_index])):
                if len(self.boxes[last_row_index][i]) > 0:
                    check_boxes_index.append((last_row_index, i))
            temp_flag = check_points_movable(check_boxes_index, ((-1, 0), (-1, -1), (-1, 1)))
            if temp_flag:
                self.boxes = self.boxes[:-1]

        # 处理第一列
        temp_flag = True
        while temp_flag:
            check_boxes_index = []
            for i in range(len(self.boxes)):
                if len(self.boxes[i][0]) > 0:
                    check_boxes_index.append((i, 0))
            temp_flag = check_points_movable(check_boxes_index, ((0, 1), (1, 1), (-1, 1)))
            if temp_flag:
                for i in range(len(self.boxes)):
                    self.boxes[i] = self.boxes[i][1:]

        # 处理最后一列
        temp_flag = True
        while temp_flag:
            check_boxes_index = []
            last_column_index = len(self.boxes[0]) - 1
            for i in range(len(self.boxes)):
                if len(self.boxes[i][last_column_index]) > 0:
                    check_boxes_index.append((i, last_column_index))
            temp_flag = check_points_movable(check_boxes_index, ((0, -1), (1, -1), (-1, -1)))
            if temp_flag:
                for i in range(len(self.boxes)):
                    self.boxes[i] = self.boxes[i][:-1]

    def check_data_losing(self):
        point_in_boxes = []
        for i in range(len(self.boxes)):
            for j in range(len(self.boxes[i])):
                if len(self.boxes[i][j]) > 1:
                    print(i, j, self.boxes[i][j])
                point_in_boxes += self.boxes[i][j]

        return len(point_in_boxes) == len(self.data)

    def output_result(self):
        """
        输出最终离散结果
        :return:
        """
        result = [0] * len(self.data)
        for r in range(len(self.boxes)):
            for c in range(len(self.boxes[0])):
                if len(self.boxes[r][c]) > 0:
                    result[self.boxes[r][c][0]] = (r, c)
        return_result = {'box_info': (len(self.boxes), len(self.boxes[0])), 'passport_info': result}
        with open('passport2matrix_result.json', 'w') as result_file:
            json.dump(return_result, result_file)
        return return_result


def euclidean_distance_for_numpy_array(point1, point2):
    """
    计算欧式距离
    :param point1:
    :param point2:
    :return: euclidean distance between two point
    """
    return np.linalg.norm(point1 - point2)


def manhattan_distance(point1, point2):
    """
    计算曼哈顿距离
    :param point1:
    :param point2:
    :return:
    """
    return sum(list(map(lambda x: abs(x[0] - x[1]), zip(point1, point2))))


def get_direction(point_axis, center_point_axis):
    """
    计算点与中心点的相对方向
    :param point_axis:
    :param center_point_axis:
    :return:
    """
    # print('getdirection',point_axis, center_point_axis)
    different = point_axis - center_point_axis
    if different[0] == 0:
        different[1] /= abs(different[1])
        return tuple(different)
    elif different[1] == 0:
        different[0] /= abs(different[0])
        return tuple(different)
    else:
        tan_angle = different[1] / different[0]
        if abs(tan_angle) <= 0.26:
            return different[0] / abs(different[0]), 0
        elif abs(tan_angle) >= 3.732:
            return 0, different[1] / abs(different[1])
        else:
            return different[0] / abs(different[0]), different[1] / abs(different[1])


def generate_reconstruct_traffic_network():

    with open('fake_passport2matrix_result_same_shape.json', 'r') as p2m_file:
        # [[r, l], [data]]
        network_info = json.load(p2m_file)

    with open('../data/efficient_passport_flow5.pickle', 'rb') as flow_file:
        # {p:[]}
        flow_info = pickle.load(flow_file)

    with open('../data/efficient_pass_port_list.json', 'r') as pp_file:
        # []
        pass_port_info = json.load(pp_file)

    reconstruct_network = np.zeros((len(flow_info[list(flow_info.keys())[0]]),
                                    network_info[0][0],
                                    network_info[0][1]))
    for box_index, box in enumerate(network_info[1]):
        pass
        reconstruct_network[:, box[0], box[1]] = flow_info[pass_port_info[box_index]]

    return reconstruct_network


def total_operation(users_l_vec):
    users_l_vec = np.array([
        [-1, 3],
        [5, 1],
        [9, -4],
        [3, 7],
        [4, 6]
    ])
    rejection = RejectionModel(users_l_vec)
    rejection.show_raw_plot()
    print(len(rejection.boxes))
    rejection.handel_all_box()
    rejection.show_boxes_with_plt(name='rejection.png', title='rejection')
    rejection.naive_shrink(4)
    rejection.show_boxes_with_plt(name='first-shrink.png', title='first shrink')
    rejection.naive_shrink(4)
    rejection.show_boxes_with_plt(name='second-shrink.png', title='shrink result')
    # print('    ', end='')
    # print(''.join(['{:^5}'.format(ii) for ii in range(len(rejection.boxes[0]))]))
    # for ii, column in enumerate(rejection.boxes):
    #     print('{:<3}:'.format(ii), end='')
    #     for e in column:
    #         print('{:^5}'.format(str(e)), end='')
    #     print()
    # rejection.plot_cliques()
    rejection.clique_shrink_after_naive_shrink()
    # print('    ', end='')
    # print(''.join(['{:^5}'.format(ii) for ii in range(len(rejection.boxes[0]))]))
    # for ii, column in enumerate(rejection.boxes):
    #     print('{:<3}:'.format(ii), end='')
    #     for e in column:
    #         print('{:^5}'.format(str(e)), end='')
    #     print()
    rejection.show_boxes_with_plt(name='cliques-shrink.png')
    print(rejection.check_data_losing())
    print(rejection.output_result())


def fake_p2m():
    """
    随机生成和passport2matrix类似的矩阵，用于对比实验
    :return:
    """
    with open('passport2matrix_result.json', 'r') as f:
        true_result_size, p2m_mapping = json.load(f)
    matrix_whole_length = true_result_size[0] * true_result_size[1]
    matrix_whole_length = list(range(matrix_whole_length))
    passport_count = len(p2m_mapping)

    fake_mapping = random.sample(matrix_whole_length, passport_count)
    fake_mapping = [(i//true_result_size[1], i % true_result_size[1]) for i in fake_mapping]

    with open('fake_passport2matrix_result.json', 'w') as f:
        json.dump((true_result_size, fake_mapping), f)


def fake_p2m_with_same_shape():
    """
    随机生成和passport2matrix类似的矩阵，点的位置都一样，但是相互关系不一样，用于对比实验
    :return:
    """
    with open('passport2matrix_result.json', 'r') as f:
        true_result_size, p2m_mapping = json.load(f)

    with open('fake_passport2matrix_result_same_shape.json', 'w') as f:
        json.dump((true_result_size, random.sample(p2m_mapping, len(p2m_mapping))), f)


if __name__ == '__main__':
    pass
    # fake_p2m_with_same_shape()

    # fake_p2m()
    total_operation("cc")

    # reconstruct_traffic_network = generate_reconstruct_traffic_network()
    # with open('../data/fake_reconstruct_same_shape_traffic_flow5.json', 'w') as data_file:
    #     json.dump(reconstruct_traffic_network.tolist(), data_file)
    # with open('../data/fake_reconstruct_same_shape_traffic_flow5_numpy.pickle', 'wb') as data_file:
    #     pickle.dump(reconstruct_traffic_network, data_file)
