import argparse
import db.DB as DB
import time
import random
import networkx as nx
import node2vec as node2vec
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
conn = DB.MysqlConn()

def select_users(params):
    if len(params) == 0:
        return False
    sql = "select retweet_uid from retweetWithoutContent where original_mid in %s"
    r = conn.select(sql, params)
    users = set(i[0] for i in r)
    return users

def select_friends(params):
    if len(params) == 0:
        return False
    sql = "select friends from weibo_network where user_id in %s"
    r = conn.select(sql, params)
    friends = [j for i in r for j in i[0].strip().split("#")]
    return friends

def select_fans_num(params):
    sql = "select followers_count from user_profile where user_id in " \
          "(select user_id from uidlist where map_id in %s)"
    r = conn.select(sql, params)
    fans_num = sum(i[0] for i in r)
    return fans_num

def write_user_graph(params):
    # 获取全部用户map_id
    users = select_users(params)
    print("所选用户群体：", users.__len__())

    # 获取用户好友，将用户节点及边存放入user_graph.txt
    s = time.time()
    edges_num = 0
    with open("user_graph.txt", "w+", encoding='utf-8') as fw:
        for i, user_id in enumerate(users, 1):
            friends = select_friends({user_id})
            for friend in friends:
                # 仅使用参与用户构造网络
                if (friend != ''):
                    edges_num += 1
                    fw.write(friend + " " + str(user_id) + "\n")
            if i % 50 == 0:
                print(i)
    print("用户边：", edges_num)
    print(time.time()-s)


def parse_args():
    parser = argparse.ArgumentParser(description="Run node2vec.")
    parser.add_argument('--input', nargs='?', default='../../resources/model/user_graph.txt', help='Input graph path')
    parser.add_argument('--output', nargs='?', default='../../resources/model/user_vec.model', help='Embeddings path')
    parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')
    parser.add_argument('--walk-length', type=int, default=20, help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walks', type=int, default=20, help='Number of walks per source. Default is 10.')
    parser.add_argument('--window-size', type=int, default=4, help='Context size for optimization. Default is 10.')
    parser.add_argument('--iter', default=10, type=int, help='Number of epochs in SGD')
    parser.add_argument('--p', type=float, default=2, help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=0.5, help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true', help='Default is weighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true', help='Default is directed.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=True)

    return parser.parse_args()

def read_graph(args):
    # 读取用户关系网络
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    if not args.directed:
        G = G.to_undirected()

    return G

def learn_embeddings(walks, args):
    # 通过使用SGD优化Skipgram目标来学习表示
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, iter=args.iter)
    model.save(args.output)

def main(args):
    nx_G = read_graph(args)
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks, args)

if __name__ == '__main__':
    # write_user_graph({3338745751776606, 3338812282191870})
    args = parse_args()
    # main(args)
    model = KeyedVectors.load(args.output)

    topn = 10
    total = [0, 0, 0]
    # users = random.sample(model.wv.index2entity, 10)
    # users = model.wv.index2entity[:10]
    users = select_users({3338745751776606, 3338812282191870})
    print(users)
    for index in users:
        if index == 326454:
            continue
        friends_num = 0
        fans_num = 0
        out_num = 0
        for similar in model.wv.most_similar(str(index), topn=topn):
            similar = similar[0]
            friends = select_friends({index})
            fans = select_friends({similar})
            f1 = similar in friends
            f2 = index in fans
            if f1 or f2:
                if f1:
                    friends_num += 1
                if f2:
                    fans_num += 1
            else:
                out_num += 1
        total[0] += friends_num
        total[1] += fans_num
        total[2] += out_num
        print("相似用户中好友占比：%s，粉丝的占比：%s，无关系的占比%s，" %
              (friends_num/topn, fans_num/topn, out_num/topn))
    print("总占比：好友%s，粉丝%s，无关系%s，" %
          (total[0]/(topn*len(users)), total[1]/(topn*len(users)), total[2]/(topn*len(users))))
    conn.close()
# 7.23事件
# 原创微博{3338745751776606, 3338812282191870}
# 本数据参与用户1461，内部边1260
# 参与用户与好友（关注者）182616，边468278（参与用户的关注总量）
# 参与用户中平均每用户320位好友
# 参与用户与粉丝（追随者）共??????，边11328861（参与用户的粉丝总量）
# 参与用户中平均每用户7754位粉丝

# 全部用户
# [795, 2472, 2811, 3266, 3621, 3765, 5402, 6534, 6627, 6691, 6861, 8469, 9082, 10307, 10735, 11203, 11947, 12074,
# 12503, 14412, 14635, 15029, 16452, 17077, 17198, 17924, 18803, 20597, 21834, 22991, 23532, 24158, 24162, 25006, 25155,
# 27200, 27679, 27943, 28023, 28170, 31716, 31865, 31923, 32172, 32384, 32469, 33665, 34874, 34894, 35113, 38342, 38791,
# 39455, 40743, 41483, 41724, 42039, 43410, 43557, 43705, 44604, 45119, 45946, 46128, 46813, 47194, 47472, 48124, 48378,
# 48444, 48569, 49255, 49554, 49696, 49889, 49941, 50277, 50398, 50671, 50896, 51288, 53219, 55671, 56271, 56311, 59340,
# 60229, 60443, 60871, 60923, 61342, 61392, 62114, 63075, 63644, 63668, 64066, 64145, 66777, 66785, 66899, 68844, 70452,
# 71076, 71403, 72208, 72927, 73726, 73943, 74611, 75058, 75683, 76617, 77107, 77405, 78182, 78300, 79790, 80737, 80818,
# 80825, 81001, 82335, 82949, 83194, 83582, 84137, 84341, 84364, 85485, 85762, 86676, 87429, 88075, 88095, 88224, 90098,
# 91150, 91612, 92056, 94489, 94687, 98615, 100822, 102572, 102853, 103798, 104400, 104436, 104460, 105520, 106931,
# 106937, 109548, 109591, 110252, 111573, 112344, 213795, 214100, 214116, 214213, 215680, 216856, 216887, 216933,
# 217416, 217602, 217818, 218982, 219712, 219749, 220746, 220767, 220879, 224027, 224371, 225459, 226088, 226343,
# 226589, 227403, 228190, 228912, 229914, 230050, 230221, 231990, 231996, 232106, 232524, 232547, 232992, 233132,
# 233252, 233900, 234552, 234709, 235198, 235600, 236199, 237386, 237869, 238247, 238365, 238909, 240219, 240923,
# 241107, 241564, 241652, 241927, 242209, 244108, 244775, 246866, 246930, 247568, 248153, 248345, 248740, 248880,
# 248900, 248991, 249157, 250649, 250832, 251093, 253152, 253440, 253816, 254380, 254533, 255152, 255743, 257087,
# 260876, 260956, 262609, 262979, 263650, 265225, 266321, 266332, 267219, 267306, 268202, 269106, 269221, 270603,
# 272428, 273790, 273806, 274604, 274686, 276802, 277492, 279461, 282927, 283165, 283309, 284901, 285520, 285757,
# 286708, 287437, 287527, 288457, 289484, 290049, 291358, 291992, 292944, 294537, 295237, 296819, 296881, 297404,
# 297880, 297939, 298411, 299247, 300532, 301089, 306056, 306819, 309775, 310736, 311307, 311963, 312104, 312246,
# 313705, 314155, 314630, 314918, 317093, 317104, 318725, 319501, 321762, 323018, 325012, 325979, 326526, 327188,
# 327903, 330423, 330816, 331963, 336230, 338007, 338808, 339575, 343891, 345783, 346156, 347048, 347170, 347414,
# 348478, 348623, 348748, 348924, 349924, 350178, 351747, 352630, 353665, 353764, 356970, 358237, 359272, 359295,
# 359319, 361039, 363722, 363905, 363973, 365166, 367099, 367528, 367826, 367952, 369331, 370277, 370875, 371530,
# 372200, 373411, 373839, 374488, 374546, 376093, 376216, 377127, 377656, 377938, 378343, 379460, 380046, 381193,
# 383102, 383728, 384113, 388742, 389799, 391546, 392681, 393290, 393698, 394195, 395201, 396072, 396517, 397674,
# 398157, 401644, 401888, 402402, 403521, 406794, 407956, 408279, 409390, 409483, 411956, 412405, 412688, 413237,
# 417895, 419245, 421849, 421952, 422216, 422741, 423422, 423839, 425991, 426258, 426266, 429134, 430038, 432850,
# 432947, 432988, 433182, 433540, 433739, 434473, 434657, 435700, 438105, 440706, 440764, 442447, 445073, 445894,
# 446038, 446068, 448400, 448414, 449495, 450029, 451478, 451495, 451681, 452564, 452824, 453500, 453893, 453963,
# 454000, 454150, 455153, 455183, 455546, 456209, 456923, 457024, 457969, 458357, 458702, 459454, 459899, 460041,
# 460211, 460612, 460805, 460850, 461956, 462493, 463141, 463405, 463930, 464601, 464743, 465242, 465330, 465453,
# 465705, 466883, 468627, 470739, 471469, 471615, 472250, 477561, 479462, 479562, 480484, 480779, 481510, 486583,
# 487687, 488013, 488651, 489187, 492854, 494241, 494420, 495811, 497460, 499095, 500132, 500615, 501157, 503515,
# 503563, 503757, 504495, 505283, 505980, 506490, 507707, 508761, 508931, 509042, 509235, 509611, 509626, 509721,
# 510783, 512169, 512956, 512983, 513112, 513320, 513521, 515645, 515710, 516444, 517103, 517742, 518460, 518706,
# 518891, 520291, 521757, 522137, 522166, 522938, 523344, 524178, 524285, 524530, 525375, 529631, 530199, 531082,
# 531208, 533103, 536162, 537344, 538731, 539989, 542769, 543730, 544363, 545878, 549775, 550386, 551324, 551749,
# 552127, 552547, 553196, 555346, 555513, 556820, 557916, 558655, 560029, 560856, 562305, 562445, 563152, 563567,
# 564111, 564224, 564991, 567417, 568519, 568782, 571375, 571399, 571724, 572720, 575360, 575534, 576115, 579350,
# 585158, 585623, 586813, 588758, 588859, 590320, 593978, 595595, 596282, 596427, 599799, 601781, 604338, 605623,
# 605827, 607309, 608775, 610130, 612045, 612843, 612883, 613643, 613813, 615610, 616542, 617490, 620097, 620685,
# 621599, 621971, 623002, 624220, 625337, 626059, 626535, 631302, 631425, 631539, 634016, 636002, 636345, 639677,
# 641643, 643057, 643250, 644327, 644413, 645445, 645474, 646447, 646861, 646906, 648585, 650045, 651271, 651334,
# 652408, 653661, 653875, 654836, 658238, 660077, 660365, 660926, 661815, 662810, 663386, 663448, 664418, 665165,
# 665656, 669122, 670040, 670692, 672698, 676523, 676546, 677782, 677930, 678786, 679542, 685250, 685333, 685659,
# 687799, 688715, 691224, 692414, 693551, 694670, 697896, 698317, 700127, 701806, 705284, 706197, 707701, 708886,
# 709447, 713782, 716478, 717108, 718367, 718671, 718893, 718917, 722675, 723198, 724233, 728312, 730865, 732837,
# 734198, 741649, 741826, 743808, 743880, 744949, 750806, 751109, 752079, 752469, 753359, 790685, 791270, 793058,
# 794364, 794456, 796781, 796965, 797381, 797736, 797865, 797945, 797976, 798779, 799693, 799855, 800480, 801086,
# 802982, 803266, 804270, 804979, 806230, 806259, 807677, 808062, 808068, 808124, 808243, 808845, 809399, 810377,
# 811181, 811458, 812076, 812199, 812986, 813284, 814342, 814625, 814638, 817065, 818307, 818491, 818564, 819368,
# 819992, 821127, 821560, 821570, 822026, 822494, 823268, 824363, 826140, 826347, 826868, 827463, 828702, 830405,
# 830582, 831951, 831955, 835763, 838218, 838325, 838889, 840018, 840143, 840935, 843437, 845603, 846713, 847359,
# 848153, 848444, 848654, 850311, 851680, 853477, 854076, 854343, 854556, 854808, 855608, 855996, 856086, 856300,
# 856363, 856442, 856584, 856586, 856705, 857430, 857776, 857830, 859200, 860239, 860407, 860621, 860774, 861002,
# 861138, 861291, 861866, 861872, 861910, 862990, 864880, 865233, 867485, 868265, 868304, 869159, 869228, 870469,
# 872052, 872518, 872772, 872802, 873596, 873657, 875208, 875523, 875879, 876664, 876672, 876967, 876995, 877336,
# 877574, 878471, 878611, 879896, 880577, 882287, 884102, 884236, 886671, 887719, 888021, 889991, 890507, 890790,
# 891262, 892804, 893235, 893907, 893934, 895052, 895363, 895947, 896555, 896693, 896754, 897241, 897571, 898252,
# 899009, 899286, 900094, 900666, 902213, 902805, 902968, 903889, 904082, 904550, 906124, 906167, 908009, 908438,
# 908614, 908665, 908754, 909700, 910360, 910404, 910734, 911057, 913325, 913917, 914110, 915151, 915304, 916584,
# 916743, 917389, 917457, 917962, 918097, 918171, 918203, 918774, 919411, 919698, 920636, 920924, 922724, 925788,
# 929701, 929973, 930231, 930234, 930618, 931208, 931695, 932427, 932555, 935317, 936056, 937301, 938587, 939389,
# 940482, 941285, 941375, 943032, 944039, 945984, 948458, 949363, 949365, 951072, 951478, 951692, 952561, 953382,
# 955578, 956206, 956255, 956938, 956966, 957296, 957862, 958006, 959161, 959209, 960926, 963951, 964629, 964933,
# 966107, 967842, 970670, 971460, 971471, 971621, 972494, 972534, 977331, 979839, 980589, 983327, 990344, 990437,
# 990681, 991247, 991337, 991654, 995389, 996326, 996764, 999045, 999649, 1000679, 1002019, 1002039, 1002489, 1002855,
# 1002931, 1006440, 1006540, 1011708, 1011798, 1012040, 1013514, 1013525, 1014184, 1014219, 1014911, 1014928, 1015354,
# 1015673, 1016861, 1019414, 1019531, 1021521, 1022399, 1022546, 1026264, 1027203, 1027954, 1029551, 1030198, 1031426,
# 1031990, 1033359, 1035176, 1035623, 1036892, 1037928, 1038877, 1039753, 1042700, 1042811, 1043650, 1044910, 1046888,
# 1047680, 1048200, 1048315, 1048476, 1049402, 1050420, 1052771, 1053076, 1053508, 1055050, 1056035, 1056540, 1058962,
# 1060919, 1061313, 1062619, 1064965, 1065307, 1065729, 1066531, 1067743, 1068067, 1069373, 1070973, 1071749, 1071810,
# 1072613, 1073131, 1073577, 1075123, 1075740, 1080418, 1082604, 1083164, 1083273, 1083339, 1084286, 1086846, 1087399,
# 1087513, 1088556, 1089464, 1089571, 1092629, 1096377, 1098442, 1098618, 1098780, 1099608, 1102480, 1109402, 1115890,
# 1116368, 1117195, 1123601, 1128177, 1128811, 1132862, 1133628, 1134938, 1142947, 1145488, 1148204, 1148833, 1149969,
# 1150102, 1150180, 1151663, 1151916, 1155145, 1158444, 1159383, 1159984, 1163762, 1165615, 1188428, 1195026, 1198075,
# 1201846, 1204404, 1206179, 1206560, 1207342, 1208969, 1209049, 1209418, 1211508, 1212616, 1212836, 1213465, 1215643,
# 1219502, 1220068, 1224169, 1224891, 1226161, 1227780, 1227947, 1228044, 1228954, 1229439, 1231958, 1233074, 1233171,
# 1234310, 1234504, 1235280, 1237625, 1238177, 1238222, 1238636, 1239218, 1241017, 1242500, 1242849, 1243542, 1244597,
# 1245392, 1247435, 1247647, 1248623, 1248728, 1250986, 1251577, 1251663, 1252623, 1253941, 1255976, 1256159, 1263639,
# 1265064, 1270602, 1270991, 1271180, 1273496, 1274223, 1275617, 1276660, 1277803, 1279676, 1280045, 1280395, 1280959,
# 1283560, 1283991, 1284261, 1284625, 1284946, 1287137, 1287371, 1287647, 1288568, 1289612, 1290139, 1290384, 1293574,
# 1294704, 1296465, 1296466, 1298777, 1299542, 1299714, 1299951, 1300805, 1300978, 1302400, 1302548, 1304244, 1304927,
# 1305305, 1306034, 1306698, 1307367, 1307587, 1307808, 1308252, 1309337, 1310072, 1313175, 1313830, 1315544, 1316607,
# 1319853, 1322290, 1327304, 1327853, 1328044, 1336316, 1342411, 1346546, 1346592, 1354889, 1355500, 1355783, 1360598,
# 1361303, 1361890, 1362540, 1366702, 1370758, 1372816, 1373175, 1374278, 1374493, 1383727, 1384468, 1450156, 1452569,
# 1452807, 1453058, 1453168, 1454163, 1455191, 1455252, 1455832, 1456429, 1457317, 1457519, 1458286, 1458558, 1458834,
# 1460047, 1460535, 1460963, 1462104, 1462105, 1463055, 1463485, 1463510, 1466651, 1467185, 1467609, 1468722, 1469763,
# 1471306, 1471799, 1473967, 1474351, 1474493, 1476364, 1479104, 1480640, 1481006, 1483500, 1483598, 1483941, 1484361,
# 1484565, 1485036, 1485434, 1487036, 1488163, 1488555, 1488633, 1488661, 1488870, 1490086, 1491241, 1492106, 1492171,
# 1492486, 1493007, 1493799, 1494009, 1494614, 1495109, 1495592, 1495674, 1497154, 1498150, 1498640, 1499282, 1501617,
# 1501931, 1501980, 1501995, 1502102, 1502352, 1503623, 1504057, 1504166, 1504754, 1504794, 1504918, 1505127, 1505671,
# 1507981, 1508947, 1510703, 1510854, 1511268, 1511328, 1512876, 1513548, 1513726, 1514229, 1514244, 1514275, 1514393,
# 1515074, 1515829, 1516194, 1516383, 1516441, 1523796, 1524693, 1527407, 1533240, 1533511, 1534431, 1534617, 1534736,
# 1535941, 1537312, 1537838, 1538106, 1538860, 1539259, 1539391, 1544670, 1545938, 1546562, 1548018, 1548649, 1549551,
# 1549788, 1549815, 1550012, 1553359, 1554172, 1554205, 1554659, 1554745, 1557759, 1559086, 1560228, 1561904, 1563606,
# 1565914, 1567291, 1577992, 1578973, 1579996, 1581553, 1584317, 1584552, 1588125, 1588157, 1589568, 1591444, 1591552,
# 1591856, 1592088, 1592108, 1593013, 1599576, 1600491, 1602026, 1604744, 1605082, 1605823, 1606587, 1608641, 1609192,
# 1609492, 1609832, 1611612, 1611647, 1612149, 1613501, 1613735, 1614778, 1614956, 1616226, 1617128, 1617756, 1617762,
# 1618441, 1620003, 1620337, 1624335, 1625292, 1626442, 1629136, 1629885, 1630199, 1630927, 1635497, 1635624, 1636527,
# 1636611, 1637580, 1639246, 1639727, 1640329, 1641647, 1644086, 1645261, 1647494, 1649127, 1651878, 1651883, 1652042,
# 1656377, 1657646, 1660222, 1661434, 1662454, 1662741, 1663584, 1664147, 1665481, 1668614, 1669613, 1670634, 1673139,
# 1673298, 1678056, 1678788, 1681701, 1686396, 1691963, 1693786, 1693878, 1695705, 1696964, 1697457, 1698844, 1698949,
# 1699780, 1699998, 1701648, 1703684, 1705537, 1706579, 1707062, 1707455, 1708204, 1711854, 1711940, 1714399, 1716136,
# 1718367, 1719800, 1724113, 1726540, 1727677, 1729512, 1730537, 1738047, 1738383, 1738665, 1739044, 1741066, 1742568,
# 1748089, 1749217]
