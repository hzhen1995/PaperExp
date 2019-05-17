import main.DB as DB
import time


def search(conn):
    sql = "select * from root_content"
    original_list = conn.select(sql, [])
    count = 0
    for original in original_list:
        if ("谣言" in original[1])|("辟谣" in original[1])|("造谣" in original[1]):
            count += 1
            print(original[0], original[4], original[5], original[6], original[1])
    print(count)

def searchRumor(conn):
    sql0 = "SELECT * from root_content where  content like %s"
    sql1 = "SELECT * from uidlist WHERE map_id = %s"
    sql2 = "SELECT * from user_profile WHERE user_id = %s"
    sql3 = "SELECT * from root_content WHERE original_uid = %s"
    sql4 = "SELECT * from root_content WHERE original_mid = %s"
    original_list = conn.select(sql0, ["%韩寒%"])
    for original in original_list:
        print(original)


if __name__ == "__main__":
    conn = DB.MysqlConn()
    search(conn)
    # searchRumor(conn)
    conn.close()