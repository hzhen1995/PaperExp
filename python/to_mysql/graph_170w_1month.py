import db.DB as DB
import time


def insertDB(conn):
    with open("../../resources/graph_170w_1month.txt", "r") as fr:
        sql = "insert into graph_temp (user_id, friends_id, timestamp, follows_count) VALUES (%s, %s, %s, %s)"
        value_list = []
        next(fr)
        pre_info = [0, "3", 1, 1]
        for count, line in enumerate(fr, 1):
            user = line.strip().split(" ")
            if (int(user[0]) == pre_info[0]) and (int(user[2]) == pre_info[2]):
                pre_info[1] += "#" + user[1]
                pre_info[3] += 1
            else:
                value_list.append(tuple(pre_info))
                pre_info = [int(user[0]), user[1], int(user[2]), 1]

            if len(value_list) == 50000:
                print(count)
                start = time.time()
                conn.executeMany(sql, value_list)
                value_list = []
                print("内部", time.time() - start)
        start = time.time()
        print(count)
        value_list.append(tuple(pre_info))
        conn.executeMany(sql, value_list)
        print("外部", time.time() - start)


def transformDB(conn):
    sql_insert = "insert into graph_170w_1month (user_id, friends_id, timestamp, follows_count) " \
                 "VALUES (%s, %s, %s, %s)"
    value_list = []
    for count, user_id in enumerate(range(1787443), 1):
        sql_select = "select user_id, friends_id, timestamp, follows_count from graph_temp " \
                     "where user_id=%s asc timestamp"
        result = conn.select(sql_select, [user_id])
        pre_info = list(result[0])
        for i in range(1, len(result)):
            if result[i][2] == pre_info[2]:
                pre_info[1] += "#" + result[i][1]
                pre_info[3] += result[i][3]
            else:
                value_list.append(tuple(pre_info))
                pre_info = list(result[i])

        value_list.append(tuple(pre_info))
        if len(value_list) > 50000:
            print(count)
            start = time.time()
            conn.executeMany(sql_insert, value_list)
            value_list = []
            print("内部", time.time() - start)
    print(count)
    start = time.time()
    conn.executeMany(sql_insert, value_list)
    print("外部", time.time() - start)


if __name__ == "__main__":
    start = time.time()
    conn = DB.MysqlConn()
    # insertDB(conn)
    # transformDB(conn)
    conn.close()
    print(time.time() - start)

# 共423347904条记录
