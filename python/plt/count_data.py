import numpy as np
import dao.social_networks_dao as snd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['xtick.direction'] = "in"
matplotlib.rcParams['ytick.direction'] = "in"
plt.figure(figsize=(5.2, 5.2))
plt.rcParams['savefig.dpi'] = 1000
plt.rcParams['figure.dpi'] = 1000

# ans = snd.get_renum_by_original_mid(["3339187067795316", "3338460728295803"])
# ans = snd.get_renum_by_original_mid(["3409823347274485", "3405653819760021"])
ans = snd.get_renum_by_original_mid(["3486534206012960", "3486542481846477"])
x = range(4, (len(ans[0]) + 1) * 4, 4)
x1 = [i - 0.8 for i in x]
x2 = [i + 0.8 for i in x]
plt.bar(x1, ans[0], width=1.6, edgecolor="black", color='#FFB5C5', label="rumor retweet number")
plt.bar(x2, ans[1], width=1.6, edgecolor="black", color='#7EC0EE', label="anti-rumor retweet number")
plt.axis([0, 44, 0, 150])
plt.legend()
plt.xticks(range(4, (len(ans[0]) + 1) * 4, 4))
plt.xlabel('time', fontsize=12)
plt.ylabel('number of retweet', fontsize=12)
plt.show()
