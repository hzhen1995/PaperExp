import db.DB as DB
import time


def insertDB(conn):
    with open("../resources/retweetWithoutContent/total.txt", "r") as fr:
        sql = "insert into retweetWithoutContent (original_mid, retweet_uid, retweet_time) VALUES (%s, %s, %s)"
        count = 0
        flag = True
        values_list = list()
        one_retweet = list()
        original_mid = ""
        for line in fr:
            one = line.strip().split(" ")
            if flag:
                original_mid = one[0]
                flag = False
            else:
                one_retweet.append(original_mid)
                for i in range(len(one)):
                    one_retweet.append(one[i])
                    if i%2==1:
                        values_list.append(one_retweet)
                        one_retweet = [original_mid]
                one_retweet = list()
                flag = True
            count += 1
            if len(values_list) > 1000000:
                print(count)
                start = time.time()
                conn.executeMany(sql, values_list)
                values_list = list()
                print(time.time() - start)
        print(count)
        start = time.time()
        conn.executeMany(sql, values_list)
        print(time.time() - start)

def updateDB(conn):
    with open("../resources/retweetWithoutContent/total.txt", "r") as fr:
        sql = "update root_content set original_uid=%s, original_time=%s, retweet_num=%s where original_mid=%s"
        count = 0
        flag = True
        values_list = list()
        for line in fr:
            if flag:
                original = line.strip().split(" ")
                one_retweet = [original[2], original[1], original[3], original[0]]
                values_list.append(one_retweet)
                flag = False
            else:
                flag = True
            count += 1
            if len(values_list) == 500:
                print(count)
                start = time.time()
                conn.executeMany(sql, values_list)
                values_list = list()
                print(time.time() - start)
        print(count)
        start = time.time()
        conn.executeMany(sql, values_list)
        print(time.time() - start)

def insertFile():
    with open("../resources/retweetWithoutContent/total.txt", "r") as fr:
        count = 0
        flag = True
        values_list = list()
        original = list()
        for line in fr:
            if flag:
                original = line.strip().split(" ")
                flag = False
            else:
                one = line.strip().split(" ")
                original.append(str(int(len(one)/2)))
                values_list.append(original)
                flag = True
            count += 1
        print(count)
        print(len(values_list))
        with open("../resources/root_content/other_message.txt", "w+", encoding="gbk") as fw:
            for original in values_list:
                line = original[0]+" "+original[1]+" "+original[2]+" "+original[3]+" "+original[4]+"\n"
                fw.write(line)

def readFile():
    with open("../resources/root_content/other_message.txt", "r", encoding="gbk") as fr:
        original_dic = dict()
        for original in fr:
            original = original.strip().split(" ")
            original_dic[original[0]] = original

if __name__ == "__main__":
    start = time.time()
    # conn = DB.MysqlConn()
    # insertDB(conn)
    # updateDB(conn)
    # conn.close()
    # insertFile()
    readFile()
    print(time.time() - start)

# 共 232978 条原创微博33307190条无内容转发记录