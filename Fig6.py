# import matplotlib.pyplot as plt
# import matplotlib
import numpy as np
# import matplotlib.ticker as ticker
#
# _, ax = plt.subplots()
#
# # Be sure to only pick integer tick locations.
#
# ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
#
# y = np.array([[2.33,2.14,2.07,2.05],
#               [2.16,2.08,2.02,2.02],
#               [1.965,1.90,1.863,1.833],
#               [1.007,0.969,0.94,0.922]
#               ])
#
# label=['DDQN','DQN','FP','MAXP']
# # label=['e-2','e-3','e-4','e_decay']
# color = ['orangered', 'blue', 'deepskyblue', 'lime']
# marker = ['s','o','^','x']
#
# linestyle = ['--',':', '-.', '-', '--']
#
# p = list()
# # for k in range(4):
# #     p_temp, = plt.plot(list(range(1,5)), y[k], color = color[k], linestyle = linestyle[k],marker=marker[k],label = label[k])
# #     p.append(p_temp)
# #
# # plt.legend(loc = 4,prop={'size': 12})
# # plt.xlabel('number of UEs each cell')
# #
# # plt.ylabel('Average rate (bps)')
# # plt.ylabel('avg_rate(bps)')
#
# # fig = matplotlib.pyplot.gcf()
# # fig.set_size_inches(8.5, 5.5)
#
# # plt.grid()
# # plt.show()


import matplotlib.pyplot as plt
plt.figure(4)
data =  [     [2.33,2.23,2.15,2.05],
              [2.16,2.05,2.02,1.9],
              [1.965,1.90,1.863,1.833],
              [1.007,0.969,0.94,0.922]
        ]
# data = np.array(data).T
name_list = ['N=25','N=36','N=49','N=64']
num = len(data[0])
set_size = len(data)
total_width = 3
width = total_width / num
label=['DDQN','DQN','FP','MAXP']
color = ['maroon', 'orange', 'lawngreen', 'dodgerblue', 'olive']
x = np.linspace(0, 16, num = num, dtype = np.int32)
plt.xticks(x+total_width/2, name_list)
bar = list()
for k in range(set_size):
    bar.append(plt.bar(x, data[k], width = width, label = label[k],fc = color[k]))
    x = x + width

plt.legend(loc = 4)
plt.xlabel('number of cell')
plt.ylabel('Average rate (bps)')
plt.show()