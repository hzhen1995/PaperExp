import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import KeyedVectors
from main import rejection

def get_data(path):
    model = KeyedVectors.load(path)
    uid_list = model.wv.most_similar('1064965', topn=256)
    vec_list = []
    for i in uid_list:
        vec_list.append(model[i[0]])
    vec_list = np.array(vec_list)
    return vec_list

def reduce_data(users_h_vec):
    t_sne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    users_l_vec = t_sne.fit_transform(users_h_vec)
    return users_l_vec

def main():
    users_h_vec = get_data('../../resources/model/user_vec.model')
    users_l_vec = reduce_data(users_h_vec)
    img = rejection.total_operation(users_l_vec)
    print(type(img))
    plt.scatter(users_l_vec[:, 0], users_l_vec[:, 1])
    plt.show()


if __name__ == '__main__':
    pass
    # main()
    density = 6.686557381586809
    print(density)
    c = int(math.sqrt(3))
    print(c)

