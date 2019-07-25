from typing import List, Set, Dict

import db.DB as DB


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
    mul_friends：key为用户map_id, value（列表类型）为用户好友map_id
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

