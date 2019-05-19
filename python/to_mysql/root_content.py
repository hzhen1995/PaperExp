import db.DB as DB
import os
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
    with open("../../resources/total.txt", "r") as fr:
        sql = "update root_content set original_uid=%s, original_time=%s, retweet_num=%s, re_without_num=%s " \
              "where original_mid=%s"
        values_list = []
        for count, line in enumerate(fr, 1):
            original_info = line.strip().split(" ")
            re_without_num = len(fr.readline().strip().split(" ")) / 2
            one_retweet = [original_info[2], original_info[1], original_info[3], re_without_num, original_info[0]]
            values_list.append(one_retweet)

            if (count % 500 == 0) or (count == 232978):
                print(count)
                start = time.time()
                conn.executeMany(sql, values_list)
                values_list = []
                print(time.time() - start)


def update_DB_copy(conn):
    files = os.listdir("../../resources/retweetWithContent")
    sql_update = "update root_content set original_uid=%s, original_time=%s, retweet_num=%s, " \
                 "re_with_num=re_with_num+%s where original_mid=%s"
    sql_select = "select * from uidlist"
    uid = conn.select(sql_select, [])
    user_dic = {}
    for i in uid:
        user_dic[i[0]] = i[1]
    for i, file in enumerate(files, 1):
        print("第 ", i, " 个文件")
        with open("../../resources/retweetWithContent/" + file, "r") as fr:
            values_list = list()
            original_info = fr.readline().strip().split("	")
            for count, line in enumerate(fr, 1):
                for re_with_num in range(2 * int(line.strip())):
                    next(fr)
                uid = -1
                if original_info[1] in user_dic.keys():
                    uid = user_dic[original_info[1]]
                values = [uid, original_info[2], original_info[3], int(line.strip()), original_info[0]]
                values_list.append(values)
                original_info = fr.readline().strip().split("	")
                if len(values_list) == 50000:
                    print(count)
                    start = time.time()
                    conn.executeMany(sql_update, values_list)
                    values_list = list()
                    print(time.time() - start)
            print(count)
            start = time.time()
            conn.executeMany(sql_update, values_list)
            print(time.time() - start)


if __name__ == "__main__":
    start = time.time()
    conn = DB.MysqlConn()
    # insertDB(conn)
    # updateDB(conn)
    conn.close()
    print(time.time() - start)

# 共300000原创微博
