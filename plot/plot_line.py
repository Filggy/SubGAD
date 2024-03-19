import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import FuncFormatter
import matplotlib.markers

import matplotlib

# 设置全局Times New Roman
plt.rc('font', family='Times New Roman')
plt.rcParams.update({"mathtext.fontset": "cm"})

# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()


label_font = {'family': 'Times New Roman',
              'weight': 'black',
              'size': 35}

legend_font = {'family': 'Times New Roman',
               'weight': 'black',
               'size': 35}

plt.figure(figsize=(30, 7), dpi=300)

# 图1-------------------------------------------------------
plt.subplot(1, 3, 1)
dims = pd.read_excel('E:\desk\plot\data\curve\pool.xls')
dims_header = dims.columns
file_name = 'pool.xlsx'.replace('.xlsx', '')
colors = ["dodgerblue", 'midnightblue', '#33CC33', '#FFA400', 'crimson', '#FF1793', '#FADB14','#ED427B']
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

plt.xlabel("(a) Pool Rate", label_font)
plt.ylabel('AUC(%)', label_font)
plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9], fontsize=30, weight='black')
plt.yticks(fontsize=30, weight='black')
ms = 15

plt.ylim(50, 105)
# plt.xlim(0,max(dims.iloc[:, 0])+min(dims.iloc[:, 0])/2)
index = np.array(dims.iloc[:, 0])


plt.yticks(ticks=[60, 70, 80, 90, 100], fontsize=28)


plt.plot(index, dims.iloc[:, 1], marker='^', ms=ms, ls='-', linewidth=3, color=colors[5], label=dims_header[1], clip_on=False)
plt.fill_between(index, dims.iloc[:, 1] - dims.iloc[:, 7], dims.iloc[:, 1] + dims.iloc[:, 7], alpha=0.3, color=colors[5])

plt.plot(index, dims.iloc[:, 2], marker='*', ms=ms, ls='-', linewidth=3, color=colors[4], label=dims_header[2], clip_on=False)
plt.fill_between(index, dims.iloc[:, 2] - dims.iloc[:, 8], dims.iloc[:, 2] + dims.iloc[:, 8], alpha=0.3, color=colors[4])

plt.plot(index, dims.iloc[:, 3], marker='8', ms=ms, ls='-', linewidth=3, color=colors[0], label=dims_header[3], clip_on=False)
plt.fill_between(index, dims.iloc[:, 3] - dims.iloc[:, 9], dims.iloc[:, 3] + dims.iloc[:, 9], alpha=0.3,color=colors[0])

plt.plot(index, dims.iloc[:, 4], marker='s', ms=ms, ls='-', linewidth=3, color=colors[3], label=dims_header[4], clip_on=False)
plt.fill_between(index, dims.iloc[:, 4] - dims.iloc[:, 10], dims.iloc[:, 4] + dims.iloc[:, 10], alpha=0.3,color=colors[3])

plt.plot(index, dims.iloc[:, 5], marker='h', ms=ms, ls='-', linewidth=3, color=colors[2], label=dims_header[5], clip_on=False)
plt.fill_between(index, dims.iloc[:, 5] - dims.iloc[:, 11], dims.iloc[:, 5] + dims.iloc[:, 11], alpha=0.3,color=colors[2])

# plt.plot(index, dims.iloc[:, 2], marker='p', ms=ms, ls='-', linewidth=5, color=colors[1], label=dims_header[2], clip_on=False)
# plt.plot(index, dims.iloc[:, 7], marker='d', ms=ms, ls='-', linewidth=5, color=colors[6], label=dims_header[7], clip_on=False)
# plt.plot(index, dims.iloc[:, 8], marker='+', ms=ms, ls='-', linewidth=5, color=colors[6], label=dims_header[8], clip_on=False)
plt.grid(linestyle='-', linewidth=1)  # 生成网格


plt.subplot(1, 3, 2)
dims = pd.read_excel('data\curve\head.xlsx')
dims_header = dims.columns
file_name = 'head.xlsx'.replace('.xlsx', '')
# colors = {'蓝色':"dodgerblue", '深蓝':'midnightblue', '橙':'olivedrab','绿': '#ff6f00', '红':'crimson','黄':'#3b8686'}
colors = ["dodgerblue", 'midnightblue', '#33CC33', '#FFA400', 'crimson', '#FF1793', '#FADB14','#ED427B']
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

plt.xlabel("(b) Heads", label_font)
plt.ylabel('AUC(%)', label_font)
plt.xticks([2, 4, 6, 8], fontsize=30, weight='black')
plt.yticks(fontsize=30, weight='black')


plt.ylim(50, 105)

index = np.array(dims.iloc[:, 0])


# plt.plot(index, dims.iloc[:, 1], marker='^', ms=ms, ls='-', linewidth=5, color=colors[5], label=dims_header[1], clip_on=False)
# plt.plot(index, dims.iloc[:, 6], marker='*', ms=ms, ls='-', linewidth=5, color=colors[4], label=dims_header[6], clip_on=False)
# plt.plot(index, dims.iloc[:, 3], marker='8', ms=ms, ls='-', linewidth=5, color=colors[0], label=dims_header[3], clip_on=False)
# plt.plot(index, dims.iloc[:, 4], marker='s', ms=ms, ls='-', linewidth=5, color=colors[3], label=dims_header[4], clip_on=False)
# plt.plot(index, dims.iloc[:, 5], marker='h', ms=ms, ls='-', linewidth=5, color=colors[2], label=dims_header[5], clip_on=False)
# plt.plot(index, dims.iloc[:, 2], marker='p', ms=ms, ls='-', linewidth=5, color=colors[1], label=dims_header[2], clip_on=False)
# plt.plot(index, dims.iloc[:, 7], marker='d', ms=ms, ls='-', linewidth=5, color=colors[6], label=dims_header[7], clip_on=False)
# plt.plot(index, dims.iloc[:, 7], marker='d', ms=ms, ls='-', linewidth=5, color=colors[6], label=dims_header[7], clip_on=False)
plt.plot(index, dims.iloc[:, 1], marker='^', ms=ms, ls='-', linewidth=3, color=colors[5], label=dims_header[1], clip_on=False)
plt.fill_between(index, dims.iloc[:, 1] - dims.iloc[:, 7], dims.iloc[:, 1] + dims.iloc[:, 7], alpha=0.3, color=colors[5])

plt.plot(index, dims.iloc[:, 2], marker='*', ms=ms, ls='-', linewidth=3, color=colors[4], label=dims_header[2], clip_on=False)
plt.fill_between(index, dims.iloc[:, 2] - dims.iloc[:, 8], dims.iloc[:, 2] + dims.iloc[:, 8], alpha=0.3, color=colors[4])

plt.plot(index, dims.iloc[:, 3], marker='8', ms=ms, ls='-', linewidth=3, color=colors[0], label=dims_header[3], clip_on=False)
plt.fill_between(index, dims.iloc[:, 3] - dims.iloc[:, 9], dims.iloc[:, 3] + dims.iloc[:, 9], alpha=0.3,color=colors[0])

plt.plot(index, dims.iloc[:, 4], marker='s', ms=ms, ls='-', linewidth=3, color=colors[3], label=dims_header[4], clip_on=False)
plt.fill_between(index, dims.iloc[:, 4] - dims.iloc[:, 10], dims.iloc[:, 4] + dims.iloc[:, 10], alpha=0.3,color=colors[3])

# plt.plot(index, dims.iloc[:, 5], marker='h', ms=ms, ls='-', linewidth=3, color=colors[2], label=dims_header[5], clip_on=False)
# plt.fill_between(index, dims.iloc[:, 5] - dims.iloc[:, 11], dims.iloc[:, 5] + dims.iloc[:, 11], alpha=0.3,color=colors[2])
plt.plot(index, dims.iloc[:, 6], marker='h', ms=ms, ls='-', linewidth=3, color=colors[2], label=dims_header[6], clip_on=False)
plt.fill_between(index, dims.iloc[:, 6] - dims.iloc[:, 12], dims.iloc[:, 6] + dims.iloc[:, 12], alpha=0.3,color=colors[2])


plt.yticks(ticks=[60, 70, 80, 90, 100],  fontsize=30)

plt.grid(linestyle='-', linewidth=1)  # 生成网格



# 图3-------------------------------------------------------

plt.subplot(1, 3, 3)
dims = pd.read_excel('E:/desk/plot/data/curve/alpha.xlsx')
dims_header = dims.columns
file_name = 'alpha.xlsx'.replace('.xlsx', '')
colors = ["dodgerblue", 'midnightblue', '#33CC33', '#FFA400', 'crimson', '#FF1793', '#FADB14','#ED427B']
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

plt.xlabel("(c) Balance Rate", label_font)
plt.ylabel('AUC(%)', label_font)
plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9], fontsize=30, weight='black')
plt.yticks(fontsize=30, weight='black')

plt.ylim(50, 105)
# plt.xlim(0,max(dims.iloc[:, 0])+min(dims.iloc[:, 0])/2)
index = np.array(dims.iloc[:, 0])


plt.yticks(ticks=[60, 70, 80, 90, 100], fontsize=28)


plt.plot(index, dims.iloc[:, 1], marker='^', ms=ms, ls='-', linewidth=3, color=colors[5], label=dims_header[1], clip_on=False)
plt.fill_between(index, dims.iloc[:, 1] - dims.iloc[:, 7], dims.iloc[:, 1] + dims.iloc[:, 7], alpha=0.3, color=colors[5])

plt.plot(index, dims.iloc[:, 2], marker='*', ms=ms, ls='-', linewidth=3, color=colors[4], label=dims_header[2], clip_on=False)
plt.fill_between(index, dims.iloc[:, 2] - dims.iloc[:, 8], dims.iloc[:, 2] + dims.iloc[:, 8], alpha=0.3, color=colors[4])

plt.plot(index, dims.iloc[:, 3], marker='8', ms=ms, ls='-', linewidth=3, color=colors[0], label=dims_header[3], clip_on=False)
plt.fill_between(index, dims.iloc[:, 3] - dims.iloc[:, 9], dims.iloc[:, 3] + dims.iloc[:, 9], alpha=0.3,color=colors[0])

plt.plot(index, dims.iloc[:, 4], marker='s', ms=ms, ls='-', linewidth=3, color=colors[3], label=dims_header[4], clip_on=False)
plt.fill_between(index, dims.iloc[:, 4] - dims.iloc[:, 10], dims.iloc[:, 4] + dims.iloc[:, 10], alpha=0.3,color=colors[3])

plt.plot(index, dims.iloc[:, 5], marker='h', ms=ms, ls='-', linewidth=3, color=colors[2], label=dims_header[5], clip_on=False)
plt.fill_between(index, dims.iloc[:, 5] - dims.iloc[:, 11], dims.iloc[:, 5] + dims.iloc[:, 11], alpha=0.3,color=colors[2])


# plt.plot(index, dims.iloc[:, 1], marker='^', ms=ms, ls='-', linewidth=3, color=colors[5], label=dims_header[1], clip_on=False)
# plt.plot(index, dims.iloc[:, 2], marker='*', ms=ms, ls='-', linewidth=3, color=colors[4], label=dims_header[2], clip_on=False)
# plt.plot(index, dims.iloc[:, 3], marker='8', ms=ms, ls='-', linewidth=3, color=colors[0], label=dims_header[3], clip_on=False)
# plt.plot(index, dims.iloc[:, 4], marker='s', ms=ms, ls='-', linewidth=3, color=colors[3], label=dims_header[4], clip_on=False)
# plt.plot(index, dims.iloc[:, 5], marker='h', ms=ms, ls='-', linewidth=3, color=colors[2], label=dims_header[5], clip_on=False)
# plt.plot(index, dims.iloc[:, 2], marker='p', ms=ms, ls='-', linewidth=5, color=colors[1], label=dims_header[2], clip_on=False)
# plt.plot(index, dims.iloc[:, 7], marker='d', ms=ms, ls='-', linewidth=5, color=colors[6], label=dims_header[7], clip_on=False)
# # plt.plot(index, dims.iloc[:, 8], marker='o', ms=ms, ls='-', linewidth=4, color=colors[7], label=dims_header[8], clip_on=False)

plt.grid(linestyle='-', linewidth=1)  # 生成网格

plt.subplots_adjust(wspace=0.4)  # 调整子图之间的水平间距

plt.legend(prop=legend_font, frameon=True, framealpha=1, handlelength=1.2, handletextpad=0.3, labelspacing=0.8, columnspacing=0.6, loc='center right',facecolor="white", ncol=7, bbox_to_anchor=(0,1.15))
# plt.legend().set_bbox_to_anchor((0, 1.15, 10, 10))

plt.savefig('fig/a1.pdf', bbox_inches='tight')
