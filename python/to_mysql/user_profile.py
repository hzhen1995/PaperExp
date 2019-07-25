import db.DB as DB
import time

user_id = set()

def insertDB(conn, path):
    sql = "insert into user_profile (user_id, bi_followers_count, city, verified, followers_count, location, " \
          "province, friends_count, name, gender, created_at, verified_type, statuses_count, description) " \
          "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    with open(path, "r") as fr:
        values_list = []
        fr.seek(192)
        for count, line in enumerate(fr, 1):
            values = [line.strip(), fr.readline().strip(), fr.readline().strip(),
                      fr.readline().strip() == str(True), fr.readline().strip(), fr.readline().strip(),
                      fr.readline().strip(), fr.readline().strip(), fr.readline().strip(), fr.readline().strip(),
                      fr.readline().strip(), fr.readline().strip(), fr.readline().strip(), fr.readline().strip()]
            next(fr)
            # 该用户已存在，过滤掉
            if values[0] in user_id:
                continue
            user_id.add(values[0])
            values_list.append(values)
            if count % 50000 == 0:
                conn.executeMany(sql, values_list)
                values_list = []
                show_str = ('[%%-%ds]' % 30) % (int(30 * count / 1655678) * "#")
                print('\r%s %d%%' % (show_str, count * 100 / 1655678), end="")
        conn.executeMany(sql, values_list)
        show_str = ('[%%-%ds]' % 30) % (int(30 * count / 1655678) * "#")
        print('\r%s %d%%' % (show_str, count * 100 / 1655678), end="")
        print()


if __name__ == "__main__":
    start = time.time()
    conn = DB.MysqlConn()
    insertDB(conn, "../../resources/basic_data/user_profile1.txt")
    insertDB(conn, "../../resources/basic_data/user_profile2.txt")
    conn.close()
    print(time.time()-start)

# user_profile1 共1008750条用户信息
# user_profile2 共672335条用户信息
# 共25407条重复
# 合并后共1655678条用户信息，总时长约630秒
