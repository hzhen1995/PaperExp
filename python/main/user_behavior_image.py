import copy
import pickle
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
        user_behavior_image[k] = image
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
                    tmp = np.array(behavior)
                    tmp = np.maximum(tmp[i - 7:i + 8, j - 7:j + 8], 0).tolist()
                    alone_user_pixel[user_pixel[i][j]].append(tmp)
                    show_str = ('[%%-%ds]' % 30) % (int(30 * count / ((len(user_pixel) - 7)*(len(user_pixel[0]) - 7))) * "#")
                    print('\r%s %d%%' % (show_str, count * 100 / ((len(user_pixel) - 7)*(len(user_pixel[0]) - 7))), end="")
    print()

    print(len(alone_user_pixel))
    data = []
    label = []
    for user, behavior_24 in alone_user_pixel.items():
        for day in range(3, 24):
            tmp = [behavior_24[day - 3], behavior_24[day - 2], behavior_24[day - 1]]
            data.append(tmp)
            label.append(behavior_24[day][7][7])
            print(behavior_24[day][7][7])

    return data, label


if __name__ == "__main__":
    user_pixel_path = "../../resources/user_pixel.pkl"
    big_data_path = "../../resources/cnn_data/big_label.pkl"
    big_label_path = "../../resources/cnn_data/big_data.pkl"

    original_mid = ["3339187067795316", "3409823347274485", "3486534206012960",
                    "3338460728295803", "3405653819760021", "3486542481846477"]
    big_data, big_label = [], []

    user_pixel = pickle.load(open(user_pixel_path, "rb"))

    for one_original in original_mid:
        # 构建24个时刻的用户行为图像
        user_behavior_image = built_user_behavior_image(user_pixel, one_original)
        data, label = built_cnn_data(user_pixel, user_behavior_image)
        big_data += data
        big_label += label
        for k, v in user_behavior_image.items():
            if k != 23:
                continue
            spread, no_spread, no_user = [], [], []
            for i in range(len(v)):
                for j in range(len(v[0])):
                    if v[i][j] == 1:
                        spread.append([j, len(v) - i])
                    elif v[i][j] == 0:
                        no_spread.append([j, len(v) - i])
                    else:
                        no_user.append([j, len(v) - i])
            plt.figure(figsize=(10, 10))
            plt.scatter(np.array(spread)[:, 0], np.array(spread)[:, 1], 60, marker=',', c="#FFBDD8")
            plt.scatter(np.array(no_spread)[:, 0], np.array(no_spread)[:, 1], 60, marker=',', c="#9EC7A0")
            plt.scatter(np.array(no_user)[:, 0], np.array(no_user)[:, 1], 60, marker=',', c="DarkSlateBlue")
            plt.show()

    pickle.dump(big_data, open(big_data_path, "wb"))
    pickle.dump(big_label, open(big_label_path, "wb"))
