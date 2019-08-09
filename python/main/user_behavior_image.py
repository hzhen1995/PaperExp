import copy
import pickle
import random
from sklearn.model_selection import train_test_split
import dao.social_networks_dao as snd
import matplotlib.pyplot as plt
import numpy as np

def built_user_behavior_image(user_pixel, original_mid):
    user_behavior = snd.get_retweet_by_original_mid(original_mid)
    user_behavior_image = {}
    history_users = set()
    for k, v in user_behavior.items():
        history_users = history_users | set(v)
        image = copy.deepcopy(user_pixel)
        for row in range(len(image)):
            for column in range(len(image[0])):
                if image[row][column] in history_users:
                    image[row][column] = 1
                elif image[row][column] != -1:
                    image[row][column] = 0
        user_behavior_image[k] = np.array(image)
    return user_behavior_image


def built_cnn_data(user_pixel, user_behavior_image):

    # 划分每个用户像素阵列，用于数据训练
    count = 0
    alone_user_pixel = {}
    for i in range(7, len(user_pixel) - 7):
        for j in range(7, len(user_pixel[0]) - 7):
            count += 1
            if user_pixel[i][j] != -1:
                alone_user_pixel[user_pixel[i][j]] = []
                for t, behavior in user_behavior_image.items():
                    tmp = np.maximum(behavior[i - 7:i + 8, j - 7:j + 8], 0).tolist()
                    alone_user_pixel[user_pixel[i][j]].append(tmp)
            show_str = ('[%%-%ds]' % 30) % (int(30 * count / ((len(user_pixel) - 14)*(len(user_pixel[0]) - 14))) * "#")
            print('\r%s %d%%' % (show_str, count * 100 / ((len(user_pixel) - 14)*(len(user_pixel[0]) - 14))), end="")
    print()
    data_0 = []
    data_1 = []
    # 叠加时间
    n = 3
    for user, behavior_24 in alone_user_pixel.items():
        for day in range(n, 24):
            temp_data = []
            for i in range(n, 0, -1):
                temp_data.append(behavior_24[day - i])
            temp_label = behavior_24[day][7][7]
            if temp_label == 1:
                data_1.append([temp_data, temp_label, user])
            elif temp_label == 0:
                data_0.append([temp_data, temp_label, user])
            else:
                print("错误了")
    data_0 = random.sample(data_0, 2000)
    data_1 = random.sample(data_1, 2000)
    data = data_1 + data_0
    random.shuffle(data)
    print("1：" + str(len(data_1)) + " 0：" + str(len(data_0)) + " 总：" + str(len(data)))
    return data


if __name__ == "__main__":
    user_pixel_path = "../../resources/user_pixel.pkl"
    train_path = "../../resources/cnn_data/big_data_train.pkl"
    test_path = "../../resources/cnn_data/big_data_test.pkl"
    original_mid = {"3339187067795316", "3409823347274485", "3486534206012960",
                    "3338460728295803", "3405653819760021", "3486542481846477"}
    big_data = []
    user_pixel = pickle.load(open(user_pixel_path, "rb"))

    for one_original in original_mid:
        # 构建24个时刻的用户行为图像
        user_behavior_image = built_user_behavior_image(user_pixel, one_original)
        data = built_cnn_data(user_pixel, user_behavior_image)
        big_data += data
        # for k, v in user_behavior_image.items():
        #     if k != 23:
        #         continue
        #     spread, no_spread, no_user = [], [], []
        #     for i in range(len(v)):
        #         for j in range(len(v[0])):
        #             if v[i][j] == 1:
        #                 spread.append([j, len(v) - i])
        #             elif v[i][j] == 0:
        #                 no_spread.append([j, len(v) - i])
        #             else:
        #                 no_user.append([j, len(v) - i])
        #     plt.figure(figsize=(10, 10))
        #     plt.scatter(np.array(spread)[:, 0], np.array(spread)[:, 1], 60, marker=',', c="#FFBDD8")
        #     plt.scatter(np.array(no_spread)[:, 0], np.array(no_spread)[:, 1], 60, marker=',', c="#9EC7A0")
        #     plt.scatter(np.array(no_user)[:, 0], np.array(no_user)[:, 1], 60, marker=',', c="DarkSlateBlue")
        #     plt.show()

    random.shuffle(big_data)
    big_data_train, big_data_test = train_test_split(big_data, test_size=0.2, random_state=1)
    # pickle.dump(big_data_train, open(train_path, "wb"))
    pickle.dump(big_data_test, open(test_path, "wb"))
    # test_data = pickle.load(open(test_path, "rb"))
