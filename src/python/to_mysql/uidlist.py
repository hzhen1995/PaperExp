import main.DB as DB
import time


def insertDB(conn):
    with open("../resources/diffusion/uidlist.txt", "r") as fr:
        sql = "insert into uidlist (user_id, map_id) VALUES (%s, %s)"
        count = 0
        values_list = list()
        for line in fr:
            values = (line, count)
            values_list.append(values)
            count += 1
            print(count)
            if len(values_list) == 50000:
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