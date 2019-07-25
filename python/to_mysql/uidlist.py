import db.DB as DB
import time


def insertDB(conn):
    with open("../../resources/basic_data/uidlist.txt", "r") as fr:
        sql = "insert into uidlist (user_id, map_id) VALUES (%s, %s)"
        values_list = list()
        for count, line in enumerate(fr, 0):
            values = (line, count)
            values_list.append(values)
            if (count % 50000 == 0) or (count == 1787442):
                conn.executeMany(sql, values_list)
                values_list = list()
                show_str = ('[%%-%ds]' % 30) % (int(30 * count / 1787443) * "#")
                print('\r%s %d%%' % (show_str, count * 100 / 1787443), end="")
if __name__ == "__main__":
    start = time.time()
    conn = DB.MysqlConn()
    insertDB(conn)
    conn.close()
    print(time.time() - start)

# 共 1787443 位用户，总时长约80秒
