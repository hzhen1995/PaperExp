import datetime
from typing import List, Set, Dict

import db.DB as DB


"""
    查询原创微博信息
    params：原创微博id
    friends：用户好友map_id
"""
def get_original_info_by_original_mid(params: str) -> List[object]:
    conn = DB.MysqlConn()
    sql = "select * from root_content where original_mid = %s"
    r = conn.select(sql, params)
    original_info = list(r[0])
    conn.close()
    return original_info

"""
    查询批量微博下的全部用户
    params：原创微博id
    users：参与微博的用户map_id
"""
def get_users_by_batch_original_mid(params: Set[str]) -> Set[int]:
    conn = DB.MysqlConn()
    sql = "select retweet_uid from retweetWithoutContent where original_mid in %s"
    r = conn.select(sql, params)
    users = set(i[0] for i in r)
    conn.close()
    return users

"""
    查询微博的一天内的转发记录
    params：原创微博id
    users：参与微博的用户map_id
"""
def get_retweet_by_original_mid(params: str) -> Dict[int, List[str]]:
    conn = DB.MysqlConn()
    original_info = get_original_info_by_original_mid(params)
    sql = "select retweet_uid, retweet_time from retweetWithoutContent " \
          "where retweet_time between %s and %s and original_mid = %s order by retweet_time asc"
    r = conn.select(sql, (original_info[3], original_info[3] + datetime.timedelta(days=1), params))
    hour = datetime.timedelta(hours=1)
    user_behavior = {i: [] for i in range(24)}
    for i in r:
        # 转发微博距微博发布多少个时
        dis_time = (i[1]-original_info[3]) // hour
        user_behavior[dis_time].append(i[0])
    conn.close()
    return user_behavior

"""
    统计
    params：原创微博id
"""
def get_renum_by_original_mid(params: List[str]) -> List[int]:
    conn = DB.MysqlConn()
    rs = []
    for mid in params:
        original_info = get_original_info_by_original_mid(mid)
        sql = "select retweet_time from retweetWithoutContent " \
              "where retweet_time between %s and %s and original_mid = %s order by retweet_time asc"

        r = conn.select(sql, (original_info[3], original_info[3] + datetime.timedelta(days=2), mid))
        user_behavior = {i: 0 for i in range(12)}
        hour = datetime.timedelta(hours=4)
        retweet_num = 0
        for i in r:
            retweet_num += 1
            # 转发微博距微博发布多少个时
            dis_time = (i[0]-original_info[3]) // hour
            user_behavior[dis_time] += 1
        rs.append(list(user_behavior.values())[:10])
    conn.close()
    return rs


"""
    查询用户的全部好友
    params：用户map_id
    friends：用户好友map_id
"""
def get_friends_by_user(params: int) -> List[str]:
    conn = DB.MysqlConn()
    sql = "select friends from weibo_network where user_id = %s"
    r = conn.select(sql, params)
    friends = [j for i in r for j in i[0].strip().split("#")]
    conn.close()
    return friends

"""
    查询批量用户的全部好友
    params：用户map_id
    user_friends：key为用户map_id, value（列表类型）为用户好友map_id
"""
def get_friends_by_batch_user(params: Set[int]) -> Dict[int, List[str]]:
    conn = DB.MysqlConn()
    sql = "select user_id, friends from weibo_network where user_id in %s"
    r = conn.select(sql, params)
    user_friends = {user[0]: user[1].split("#") for user in r}
    conn.close()
    return user_friends


"""
    查询批量用户的粉丝数量
    params：用户map_id
    fans_num：粉丝数量
"""
def get_fans_num_by_batch_user(params: Set[int]) -> int:
    conn = DB.MysqlConn()
    sql = "select followers_count from user_profile where user_id in (select user_id from uidlist where map_id in %s)"
    r = conn.select(sql, params)
    fans_num = sum(i[0] for i in r)
    conn.close()
    return fans_num

