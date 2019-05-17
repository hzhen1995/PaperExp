import db.DB as DB
import time


def insertDB(conn):
    with open("../../resources/uidlist.txt", "r") as fr:
        sql = "insert into uidlist (user_id, map_id) VALUES (%s, %s)"
        values_list = list()
        for count, line in enumerate(fr):
            values = (line, count)
            values_list.append(values)
            if len(values_list) == 50000:
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
    conn.close()
    print(time.time() - start)

# 共 1787443 位用户