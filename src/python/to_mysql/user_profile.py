import main.DB as DB
import time

def insertDB(conn, path, is_first):
    with open(path, "r") as fr:
        if is_first:
            sql = "insert into user_profile (user_id, bi_followers_count, city, verified, followers_count, location, " \
                  "province, friends_count, name, gender, created_at, verified_type, statuses_count, description) " \
                  "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        else:
            sql = "insert into user_profile2 (user_id, bi_followers_count, city, verified, followers_count, location, " \
                  "province, friends_count, name, gender, created_at, verified_type, statuses_count, description) " \
                  "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        flag = 0
        count = 0
        values_list = list()
        one_record = 0
        values = ["", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        for line in fr:
            if flag==15:
                values[one_record] = line.strip()
                one_record += 1
                if one_record==15:
                    values.pop()
                    values[3] = values[3]==str(True)
                    values_list.append(values)
                    values = ["", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
                    one_record=0
                    count += 1
                    if count % 50000 == 0:
                        print(count)
                        start = time.time()
                        conn.executeMany(sql, values_list)
                        values_list = list()
                        print(time.time() - start)
            else:
                flag+=1
        print(count)
        start = time.time()
        conn.executeMany(sql, values_list)
        print(time.time() - start)

# 合并
def merge(conn):
    sql_select = "select * from user_profile where user_id=%s"
    sql_insert = "insert into user_profile (user_id, bi_followers_count, city, verified, followers_count, location, " \
                 "province, friends_count, name, gender, created_at, verified_type, statuses_count, description) " \
                 "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    sql2 = "select * from user_profile2"
    myresult = conn.select(sql2, [])
    count = 0
    insert = 0
    values_list = list()
    for one in myresult:
        count += 1
        print(count)
        if len(conn.select(sql_select, [one[0]]))<1:
            insert += 1
            values_list.append(one)
        if len(values_list)==50000:
            start = time.time()
            conn.executeMany(sql_insert, values_list)
            values_list = list()
            print("内部", time.time() - start)
    print(count)
    start = time.time()
    conn.executeMany(sql_insert, values_list)
    print("外部", time.time() - start)
    print("插入", insert)


if __name__ == "__main__":
    start = time.time()
    conn = DB.MysqlConn()
    path = ["../resources/userProfile/user_profile1.txt", "../resources/userProfile/user_profile2.txt"]
    insertDB(conn, path[0], True)
    insertDB(conn, path[1], False)
    merge(conn)
    conn.close()
    print(time.time()-start)

# user_profile1 共1008750条用户信息
# user_profile2 共672335条用户信息
# 共25407条重复
# 合并后共1655678条用户信息
