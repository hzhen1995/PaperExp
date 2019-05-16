import main.DB as DB
import time

def insertDB(conn):
    with open("../resources/graph_170w_1month.txt", "r") as fr:
        sql = "insert into graph_temp (user_id, friends_id, follows_count, timestamp) VALUES (%s, %s, %s, %s)"
        count = 0
        value_list = list()
        friend_id_list = list()
        user_id_timestamp = [0, 1]
        for line in fr:
            oneUser = line.strip().split(" ")
            user_id = int(oneUser[0])
            friend_id = int(oneUser[1])
            timestamp = int(oneUser[2])
            if (user_id!=user_id_timestamp[0])|(timestamp!=user_id_timestamp[1]):
                friends = ""
                for one_id in friend_id_list:
                    friends += "#"+str(one_id)
                value = (user_id_timestamp[0], friends.strip("#"), len(friend_id_list), user_id_timestamp[1])
                value_list.append(value)
                friend_id_list = list()
            friend_id_list.append(friend_id)
            user_id_timestamp[0] = user_id
            user_id_timestamp[1] = timestamp
            count += 1
            if len(value_list) == 50000:
                print(count)
                start = time.time()
                conn.executeMany(sql, value_list)
                value_list = list()
                print("内部", time.time() - start)
        start = time.time()
        print(count)
        last_friends = ""
        for one_id in friend_id_list:
            last_friends += "#" + str(one_id)
        last = (user_id_timestamp[0], last_friends.strip("#"), len(friend_id_list), user_id_timestamp[1])
        value_list.append(last)
        conn.executeMany(sql, value_list)
        print("外部", time.time() - start)

def transformDB(conn):
    sql_insert = "insert into graph_170w_1month (user_id, friends_id, follows_count, timestamp) VALUES (%s, %s, %s, %s)"
    value_list = list()
    count = 0
    for user_id in range(1787443):
        sql_select = "select * from graph_temp where user_id=%s"
        value_select = [user_id]
        myresult = conn.select(sql_select, value_select)
        for timestamp in range(32):
            timestamp += 1
            value = ()
            friends_id = ""
            follows_count = 0
            for one_record in myresult:
                if one_record[4]==timestamp:
                    friends_id += "#"+one_record[2]
                    follows_count += one_record[3]
                    value = (one_record[1], friends_id.strip("#"), follows_count, timestamp)
            if len(value)>0:
                value_list.append(value)
            count += 1
            if len(value_list) > 50000:
                print(count)
                start = time.time()
                conn.executeMany(sql_insert, value_list)
                value_list = list()
                print("内部", time.time() - start)
    print(count)
    start = time.time()
    conn.executeMany(sql_insert, value_list)
    print("外部", time.time() - start)

if __name__ == "__main__":
    start = time.time()
    conn = DB.MysqlConn()
    insertDB(conn)
    transformDB(conn)
    conn.close()
    print(time.time() - start)



# 共423347905条