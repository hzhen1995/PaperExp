import pickle
import random

import dao.social_networks_dao as snd


def built_factor(users):
    users = {}

    for i, user in enumerate(users):
        print(i)
        is_exist = snd.is_in_paper(user)
        if is_exist:
            continue

        # 构造用户基本属性
        att = snd.get_att_num_by__user(user)
        # 构造用户活跃度
        retwnum = snd.get_retwnum_by_user({user})
        origNum = snd.get_origNum_by_user({user})
        act = 0.5*origNum + retwnum
        # 构造用户历史转发率
        friends = snd.get_friends_by_user({user})
        friends_retwnum = snd.get_retwnum_by_user(set(friends))
        ret = 0
        if friends_retwnum != 0:
            ret = retwnum / friends_retwnum

        # 构造好友带动力
        fri = snd.get_retwnum_by_user(set(friends))
        # 构造信息与信息关联度
        rel = random.uniform(0, 1)
        # 构造信息流行度
        pop = random.uniform(0, 1)
        snd.insertPaper([user, att, act, ret, fri, rel, pop])

if __name__ == "__main__":

    users_path = "../../resources/users.pkl"
    users = pickle.load(open(users_path, "rb"))
    built_factor(users)
    print(len(users))
