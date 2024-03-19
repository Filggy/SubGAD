import matplotlib.pyplot as plt
import numpy as np

# x_labels = ["MindReader", "Amazon", "Alibaba", "MovieLens", "Last-FM"]
x_labels = ["MUTAG", "BZR", "ER_MD", "COX2", "ENZYMES", "DD", "PROTEINS","AIDS","NCI1","MMP","ARE",'p53',"aromatase"]

y1 = [0.7705, 0.7681, 0.666, 0.7475, 0.674, 0.6635, 0.8085,0.9939,0.6082,0.6948,0.6233,0.6568,0.6368] # assembel
y2 = [0.9023, 0.8521, 0.681, 0.7995, 0.565, 0.8458, 0.8486,0.9864,0.5899,0.6837,0.6602,0.6465,0.6514]
y3 = [0.9300, 0.8807, 0.7524, 0.7949, 0.702, 0.8332, 0.8515,0.9998,0.6928,0.7479,0.6807,0.7054,0.7143]
# y4 = [0.9295, 0.8807, 0.6733, 0.7949, 0.6788, 0.7487, 0.7716,0.9989,0.6915,0.7249,0.6267,0.7054,0.7143]
# y5 = [0.9395, 0.8807, 0.7524, 0.7949, 0.7017, 0.8332, 0.8515,0.9998,0.6928,0.7479,0.6807,0.7054,0.7143]

# 0.9989,0.7018,0.7442,0.6377,0.7058,0.7143

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# X轴位置
x = np.arange(len(x_labels))
# 柱图大小
width = 0.2

legend_font = {
            'family': 'Times New Roman',
            'size': 40,
            'weight': 'black',
            }

# 创建图形
figsize = 28, 10
fig, ax = plt.subplots(figsize = figsize)
# ax.bar(x + width,   y1,width,label='SEAD w/o Substructure', color='#F4DA61',linewidth=0.7)
# ax.bar(x + width*2, y2, width, label='SEAD w/o Learnable encoding', color='#DB8479',linewidth=0.7)
# ax.bar(x + width*3, y3, width,label='SEAD', color='#6179B5',linewidth=0.7)
#BFE2BF 71A2CF F39C8E
ax.bar(x + width,   y1,width,label='GASR w/o Substructure Learning', color='#BFE2BF',linewidth=1)
ax.bar(x + width*2, y2, width, label='GASR w/o Learnable Encoding', color='#F39C8E',linewidth=0.8)
ax.bar(x + width*3, y3, width,label='GASR', color='#71A2CF',linewidth=0.8)

# ax.bar(x + width,   y1,width,label='ESGAD w/o Substructure', color='#F4DA61',linewidth=0.7)
# ax.bar(x + width*2, y2, width, label='ESGAD w/o Learnable encoding', color='#EE6E39',linewidth=0.7)
# ax.bar(x + width*3, y3, width,label='ESGAD', color='#6179B5',linewidth=0.7)

# ax.bar(x + width*4, y4, width,label='SLGAD w/o L2', color='#E6A93E',linewidth=0.7)
# ax.bar(x + width*5, y5, width,label='SLGAD', color='#4F4D34',linewidth=0.7)

# Y轴标题
# ax.set_ylabel('Recall@20', fontsize=28)
ax.set_ylabel('AUC',weight='black', fontsize=40)

plt.tick_params(labelsize=33)
# plt.xticks()
# X轴坐标显示，x + width*2 标识X轴刻度所在位置 
ax.set_xticks(x + width*2.5)
ax.set_xticklabels(x_labels,weight='black')
#plt.yticks([0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40])
plt.ylim((0.00,1.10))
plt.yticks([0,0.25,0.50,0.75,1.00],weight='black', fontsize=33)
# 显示右上角图例
ax.legend(prop=legend_font,frameon=False,handlelength=1.1,handletextpad=0.2,labelspacing=0.4,framealpha=0.8,columnspacing=0.8,borderpad=0.5,loc='center',ncol=6,bbox_to_anchor=(0.5,1.08))


ax.margins(0.03, 0.03)  # 将边距设置为负数以减少空白

# 自动调整子图参数以提供指定的填充。多数情况下没看出来区别
fig.tight_layout()

bwith = 2 #边框宽度设置为2
TK = plt.gca()#获取边框
TK.spines['bottom'].set_linewidth(bwith)#图框下边
TK.spines['left'].set_linewidth(bwith)#图框左边
TK.spines['top'].set_linewidth(bwith)#图框上边
TK.spines['right'].set_linewidth(bwith)#图框右边

plt.savefig('fig/bar/assembel_ab.pdf')
plt.show()