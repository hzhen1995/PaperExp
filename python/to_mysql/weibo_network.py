import db.DB as DB
import time


def insertDB(conn):
    with open("../../resources/basic_data/weibo_network.txt", "r") as fr:
        sql = "insert into weibo_network (map_id, friends_count, friends, bi_followers_count, bi_followers) " \
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
            values = (user[0], user[1], temp[0].strip("#"), temp[1], temp[2].strip("#"))
            values_list.append(values)
            if (count % 50000 == 0) or (count == 1787443):
                conn.executeMany(sql, values_list)
                values_list = []
                show_str = ('[%%-%ds]' % 30) % (int(30 * count / 1787443) * "#")
                print('\r%s %d%%' % (show_str, count * 100 / 1787443), end="")


if __name__ == "__main__":
    start = time.time()
    conn = DB.MysqlConn()
    insertDB(conn)
    conn.close()
    print(time.time()-start)

# 共1787443个用户，其中7636用户无关注者，总时长约2469秒
