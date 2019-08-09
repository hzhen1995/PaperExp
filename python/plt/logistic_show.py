import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['xtick.direction'] = "in"
matplotlib.rcParams['ytick.direction'] = "in"
# plt.rcParams['savefig.dpi'] = 1000
# plt.rcParams['figure.dpi'] = 1000


# 画图
# Mut_topic = [0.924, 0.913125, 0.894125, 0.9024125]
# Inf_topic = [0.883125, 0.853125, 0.69125, 0.753125]
# In_topic = [0.773125, 0.83125, 0.599125, 0.6677]
# out_topic = [0.7793125, 0.84425, 0.6308, 0.687]

# Mut_topic = [0.969, 0.9575, 0.834125, 0.8824125]
# Inf_topic = [0.9553125, 0.94823125, 0.769125, 0.753125]
# In_topic = [0.78223125, 0.78125, 0.449125, 0.6077]
# out_topic = [0.7643375, 0.788375, 0.2508, 0.427]
#
# Mut_topic = [0.949, 0.92375, 0.814125, 0.8824125]
# Inf_topic = [0.753125, 0.823125, 0.33125, 0.423125]
# In_topic = [0.8223125, 0.79125, 0.419125, 0.4977]
# out_topic = [0.93375, 0.91375, 0.7508, 0.777]
#
# x = range(1, 5)
# x1 = [i - 0.27 for i in x]
# x2 = [i - 0.09 for i in x]
# x3 = [i + 0.09 for i in x]
# x4 = [i + 0.27 for i in x]
# plt.figure(figsize=(6, 6))
# plt.bar(x1, Mut_topic, width=0.18, edgecolor="black", label="Mut")
# plt.bar(x2, Inf_topic, width=0.18, edgecolor="black", label="Inf")
# plt.bar(x3, In_topic, width=0.18, edgecolor="black", label="In factor")
# plt.bar(x4, out_topic, width=0.18, edgecolor="black", label="Out factor")
# plt.axis([0, 5, 0, 1])
# plt.legend(loc="upper center", ncol=4)
# plt.xticks(np.arange(5), ("", "accuracy", "precision", "recall", "F1"))
# plt.xlabel('Metrics', fontsize=12)
# plt.ylabel('Evaluation', fontsize=12)
# plt.show()


x = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
# plt.plot([0, 0.023, 1], [0, 0.815, 1], "-", label='Mut')
# plt.plot([0, 0.031, 1], [0, 0.655, 1], "-", label='Inf')
# plt.plot([0, 0.026, 1], [0, 0.395, 1], "-", label='In factor')
# plt.plot([0, 0.059, 1], [0, 0.535, 1], "-", label='out factor')

# plt.plot([0, 0.033, 1], [0, 0.835, 1], "-", label='Mut')
# plt.plot([0, 0.039, 1], [0, 0.685, 1], "-", label='Inf')
# plt.plot([0, 0.046, 1], [0, 0.395, 1], "-", label='In factor')
# plt.plot([0, 0.089, 1], [0, 0.485, 1], "-", label='out factor')

# plt.plot([0, 0.028, 1], [0, 0.805, 1], "-", label='Mut')
# plt.plot([0, 0.031, 1], [0, 0.455, 1], "-", label='Inf')
# plt.plot([0, 0.034, 1], [0, 0.695, 1], "-", label='In factor')
# plt.plot([0, 0.066, 1], [0, 0.635, 1], "-", label='out factor')
# plt.plot([0, 1], [0, 1], "-", color="Lightgray")
# plt.xlabel('False Positive Rate', fontsize=12)
# plt.ylabel('True Positive Rate', fontsize=12)
# plt.legend()
# plt.axis([0, 1,  0, 1])
# plt.show()