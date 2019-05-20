import db.DB as DB
import time

def select_user(args):
    sql = "select retweet_uid from retweetWithoutContent where original_mid in %s "
    r = conn.select(sql, args)
    users = set(i[0] for i in r)
    return users

def select_friend(args):
    sql_temp = ""
    if len(args) != 0:
        sql_temp += "where user_id=%s "
        for i in range(len(args) - 1):
            sql_temp += "or user_id=%s "
    sql = "select friends from weibo_network " + sql_temp
    r = conn.select(sql, args)
    return r

def save_user_graph(args):
    # 获取全部用户map_id
    users = select_user(args)
    print("所选用户群体：", users.__len__())

    # 获取用户好友，将用户节点及边存放入user_graph.txt
    s = time.time()
    edges_num = [0, 0]
    with open("user_graph.txt", "w+", encoding='utf-8') as fw:
        for i, user_id in enumerate(users, 1):
            friends = select_friend([user_id])
            friends_list = friends[0][0].strip().split("#")
            for friend in friends_list:
                if (friend != '') and (int(friend) in users):
                    edges_num[0] += 1
                    fw.write(str(user_id) + " " + friend + "\n")
                else:
                    edges_num[1] += 1
            if i % 50 == 0:
                print(i)
    print("内部边：", edges_num[0], "外部边：", edges_num[1])
    print(time.time()-s)


if __name__ == '__main__':
    conn = DB.MysqlConn()
    save_user_graph({3338745751776606, 3338812282191870})
    conn.close()
