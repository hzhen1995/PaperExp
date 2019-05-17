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

#
import time

s = set()
for i in range(2000000):
    s.add(i)

print(s.__sizeof__())

tim = time.time()
if 1900000 in s:
    print(10000000)
print(time.time()-tim)