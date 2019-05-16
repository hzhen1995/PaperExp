# import time
#
# def trans(x):
#     x = str(x)
#     if x[0]=="-":
#         y = ""
#         for i in range(len(x)):
#             if (i == (len(x) - 1)) & (str(x[i]) == "0"):
#                 continue
#             y = x[i] + y
#         y = "-"+y.strip("-")
#     else:
#         y = ""
#         for i in range(len(x)):
#
#             if (i==(len(x)-1))&(str(x[i])=="0"):
#                 continue
#             else:
#                 y = x[i]+y
#     return y
#
# if __name__ =="__main__":
#     print(time.time())
#     x = -320
#     y = trans(x)
#     print(y)
#     print(time.time())

a = [2,3,4]
print(tuple(a))
