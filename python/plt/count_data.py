import numpy as np
import dao.social_networks_dao as snd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['xtick.direction'] = "in"
matplotlib.rcParams['ytick.direction'] = "in"
# plt.rcParams['savefig.dpi'] = 1000
# plt.rcParams['figure.dpi'] = 1000

# ans = snd.get_renum_by_original_mid(["3339187067795316", "3338460728295803"])
# ans = snd.get_renum_by_original_mid(["3409823347274485", "3405653819760021"])
# ans = snd.get_renum_by_original_mid(["3486534206012960", "3486542481846477"])
# plt.figure(figsize=(5.2, 5.2))
# x = range(4, (len(ans[0]) + 1) * 4, 4)
# x1 = [i - 0.8 for i in x]
# x2 = [i + 0.8 for i in x]
# plt.bar(x1, ans[0], width=1.6, edgecolor="black", color='#FFB5C5', label="rumor retweet number")
# plt.bar(x2, ans[1], width=1.6, edgecolor="black", color='#7EC0EE', label="anti-rumor retweet number")
# plt.axis([0, 44, 0, 150])
# plt.legend()
# plt.xticks(range(4, (len(ans[0]) + 1) * 4, 4))
# plt.xlabel('time', fontsize=12)
# plt.ylabel('number of retweet', fontsize=12)
# plt.show()

#
# A = [434, 356, 97, 164, 531, 885, 845, 688, 334, 176, 63, 34]
# pred = [352, 339, 149, 197, 666, 827, 890, 742, 583, 250, 54, 22]
# x = range(2, 26, 2)
# plt.plot(x, A, "-o", Markersize=3, label='real')
# plt.plot(x, pred, "-o", Markersize=3, label='predict')
# plt.xlabel('Time Interva', fontsize=12)
# plt.xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26])
# plt.ylabel('Number of Participants', fontsize=12)
# plt.legend()
# plt.axis([0, 26,  0, 1000])
# plt.show()


# B = [947, 494, 446, 310, 217, 286, 312, 217, 179, 125, 70, 69]
# pred = [756, 409, 349, 197, 236, 327, 340, 242, 253, 150, 34, 22]
# C = [598, 89, 36, 34, 20, 4, 0, 1, 8, 17, 5, 5]
# x = range(2, 26, 2)
# plt.plot(x, B, "-o", Markersize=3, label='real')
# plt.plot(x, pred, "-o", Markersize=3, label='predict')
# plt.xlabel('Time Interva', fontsize=12)
# plt.xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26])
# plt.ylabel('Number of Participants', fontsize=12)
# plt.legend()
# plt.axis([0, 26,  0, 1000])
# plt.show()



C = [598, 89, 36, 34, 20, 4, 0, 1, 8, 17, 5, 5]
pred = [433, 209, 119, 76, 32, 11, 3, 16, 25, 14, 12, 17]
x = range(2, 26, 2)
plt.plot(x, C, "-o", Markersize=3, label='real')
plt.plot(x, pred, "-o", Markersize=3, label='predict')
plt.xlabel('Time Interva', fontsize=12)
plt.xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26])
plt.ylabel('Number of Participants', fontsize=12)
plt.legend()
plt.axis([0, 26,  0, 600])
plt.show()