import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.ticker as ticker

_, ax = plt.subplots()

# Be sure to only pick integer tick locations.

ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

y = np.array([[3.360,2.331,1.951,1.727,1.527],
              [3.176,2.167,1.781,1.560,1.386],
              [2.989,1.965,1.552,1.326,1.163],
              [2.020,1.007,0.687,0.515,0.412]])

label=['DDQN','DQN','FP','MAXP']
# label=['e-2','e-3','e-4','e_decay']
color = ['orangered', 'blue', 'deepskyblue', 'lime']
marker = ['s','o','^','x']

linestyle = ['--',':', '-.', '-', '--']

p = list()
for k in range(4):
    p_temp, = plt.plot(list(range(1,6)), y[k], color = color[k], linestyle = linestyle[k],marker=marker[k],label = label[k])
    p.append(p_temp)

plt.legend(loc = 1,prop={'size': 12})
plt.xlabel('number of SEs each cell')

plt.ylabel('Average rate (bps)')
# plt.ylabel('avg_rate(bps)')

# fig = matplotlib.pyplot.gcf()
# fig.set_size_inches(8.5, 5.5)

# plt.grid()
plt.show()
