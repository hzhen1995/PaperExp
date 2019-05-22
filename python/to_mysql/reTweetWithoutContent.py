import db.DB as DB
import time


def insertDB(conn):
    with open("../../resources/basic_data/total.txt", "r") as fr:
        sql = "insert into retweetWithoutContent (original_mid, retweet_uid, retweet_time) VALUES (%s, %s, %s)"
        values_list = []
        for count, line in enumerate(fr, 1):
            original_info = line.strip().split(" ")
            retweet_info = fr.readline().strip().split(" ")
            for i in range(0, len(retweet_info), 2):
                one_retweet = [original_info[0], retweet_info[i], retweet_info[i+1]]
                values_list.append(one_retweet)

            if (count % 10000 == 0) or (count == 232978):
                print(count)
                start = time.time()
                conn.executeMany(sql, values_list)
                values_list = []
                print(time.time() - start)


if __name__ == "__main__":
    start = time.time()
    conn = DB.MysqlConn()
    # insertDB(conn)
    # updateDB(conn)
    conn.close()
    print(time.time() - start)

# 共 232978 条原创微博33307190条无内容转发记录
