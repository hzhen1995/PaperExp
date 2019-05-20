import db.DB as DB
import time

conn = DB.MysqlConn()


def select_users(args):
    if len(args) == 0:
        return False
    sql = "select retweet_uid from retweetWithoutContent where original_mid in %s"
    r = conn.select(sql, args)
    users = set(i[0] for i in r)
    return users

def select_friends(args):
    if len(args) == 0:
        return False
    sql = "select friends from weibo_network where user_id in %s"
    r = conn.select(sql, args)
    friends = [j for i in r for j in i[0].strip().split("#")]
    return friends

def select_fans_num(args):
    sql_temp = ""
    # if len(args) != 0:
    #     sql_temp += "where user_id=%s "
    #     for i in range(len(args) - 1):
    #         sql_temp += "or user_id=%s "
    sql = "select followers_count from user_profile " + sql_temp
    r = conn.select(sql, args)
    return r

def save_user_graph(args):
    # 获取全部用户map_id
    users = select_users(args)
    print("所选用户群体：", users.__len__())

    # 获取用户好友，将用户节点及边存放入user_graph.txt
    s = time.time()
    edges_num = 0
    with open("user_graph.txt", "w+", encoding='utf-8') as fw:
        for i, user_id in enumerate(users, 1):
            friends = select_friends({user_id})
            for friend in friends:
                if friend != '':
                    edges_num += 1
                    fw.write(str(user_id) + " " + friend + "\n")
            if i % 50 == 0:
                print(i)
    print("用户边：", edges_num)
    print(time.time()-s)
save_user_graph({3338745751776606, 3338812282191870})

def user_info(args):
    # 获取全部用户map_id
    users = select_users(args)
    print("所选用户群体：", users.__len__())
    for i, user_id in enumerate(users, 1):
        friends = select_friends([user_id])
        friends_list = friends[0][0].strip().split("#")
        for friend in friends_list:
            if friend != '':
                print(str(user_id) + " " + friend + "\n")
        if i % 50 == 0:
            print(i)


if __name__ == '__main__':
    conn.close()


# 7.23事件
# 原创微博{3338745751776606, 3338812282191870}
# 本数据参与用户1461，内部边1260
# 参与用户与好友（关注）共182923，边468278
# 参与用户中平均每用户320位好友，
