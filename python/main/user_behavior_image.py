import copy
import pickle
import dao.social_networks_dao as snd
import matplotlib.pyplot as plt
import numpy as np

def built_user_behavior_image(user_pixel_path, original_mid):
    user_behavior = snd.get_retweet_by_original_mid(original_mid)
    user_behavior_image = {}
    user_pixel = pickle.load(open(user_pixel_path, "rb+"))
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

    pickle.dump(user_behavior_image, open("../../resources/user_behavior_image/" + original_mid + ".pkl", "wb+"))


def built_cnn_data():
    pass

if __name__ == "__main__":
    # 构建24个时刻的用户行为图像
    # built_user_behavior_image("../../resources/user_pixel.pkl", "3338745751776606")
    image = pickle.load(open("../../resources/user_behavior_image/3338745751776606.pkl", "rb+"))
    spread, no_spread, no_user = [], [], []
    for i in range(len(image[23])):
        for j in range(len(image[23][0])):
            print(image[23][i][j])
            if image[23][i][j] == 1:
                spread.append([j, len(image[23]) - i])
            elif image[23][i][j] == 0:
                no_spread.append([j, len(image[23]) - i])
            else:
                no_user.append([j, len(image[23]) - i])
    plt.figure(figsize=(4.1, 4.1))
    plt.scatter(np.array(spread)[:, 0], np.array(spread)[:, 1], 60, marker=',', c="#FFBDD8")
    plt.scatter(np.array(no_spread)[:, 0], np.array(no_spread)[:, 1], 60, marker=',', c="#9EC7A0")
    plt.scatter(np.array(no_user)[:, 0], np.array(no_user)[:, 1], 60, marker=',', c="DarkSlateBlue")
    plt.show()

