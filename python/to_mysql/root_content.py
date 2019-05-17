import db.DB as DB
import time


def insertDB(conn):
    with open("../../resources/root_content.txt", "r") as fr:
        sql = "insert into root_content (original_mid, content) VALUES (%s, %s)"
        values_list = []
        for count, line in enumerate(fr, 1):
            one_info = [line.strip(), fr.readline().strip()]
            values_list.append(one_info)
            if (count % 10000 == 0) or (count == 300000):
                print(count)
                start = time.time()
                conn.executeMany(sql, values_list)
                values_list = []
                print(time.time() - start)


def updateDB(conn):
    with open("../resources/root_content/other_message2.txt", "r") as fr:
        sql = "update root_content set original_uid=%s, original_time=%s, retweet_num=%s, re_with_num=%s, re_without_num=%s where original_mid=%s"
        count = 0
        values_list = list()
        for line in fr:
            original = line.strip().split(" ")
            one_retweet = [original[2], original[1], original[3], original[4], original[5], original[0]]
            values_list.append(one_retweet)
            count += 1
            if len(values_list) == 10000:
                print(count)
                start = time.time()
                conn.executeMany(sql, values_list)
                values_list = list()
                print(time.time() - start)
        print(count)
        start = time.time()
        conn.executeMany(sql, values_list)
        print(time.time() - start)


if __name__ == "__main__":
    start = time.time()
    conn = DB.MysqlConn()
    insertDB(conn)
    # updateDB(conn)
    conn.close()
    print(time.time() - start)

# 共300000原创微博
