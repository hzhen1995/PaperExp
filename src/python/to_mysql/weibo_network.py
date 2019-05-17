import db.DB as DB
import time


def insertDB(conn):
    with open("../../resources/weibo_network.txt", "r") as fr:
        sql = "insert into weibo_network (user_id, friends_count, friends, bi_followers_count, bi_followers) " \
              "VALUES (%s, %s, %s, %s, %s)"
        next(fr)
        values_list = []
        for count, line in enumerate(fr, 1):
            # user = [int(x) for x in line.strip().split("	")]
            user = line.strip().split("	")
            temp = ["", 0, ""]
            for i in range(2, len(user), 2):
                temp[0] += "#"+user[i]
                if user[i+1] == "1":
                    temp[1] += 1
                    temp[2] += "#"+user[i]
            values = (int(user[0]), int(user[1]), temp[0].strip("#"), temp[1], temp[2].strip("#"))
            values_list.append(values)
            if count % 50000 == 0:
                print(count)
                start = time.time()
                conn.executeMany(sql, values_list)
                values_list = []
                print(time.time() - start)
        print(count)
        start = time.time()
        conn.executeMany(sql, values_list)
        print(time.time() - start)


if __name__ == "__main__":
    start = time.time()
    conn = DB.MysqlConn()
    insertDB(conn)
    conn.close()
    print(time.time()-start)

# 共1787443个用户，其中7636用户无关注者
