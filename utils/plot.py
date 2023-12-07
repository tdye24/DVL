import json
from constants import PID_2_NAME
import matplotlib.pyplot as plt
import numpy as np
path = '../ImageNet.json'
with open(f'./{path}', 'r') as f:
    ImageNet_res = json.load(f)
path = '../Main[20]_Target[0]_19.json'
with open(f'./{path}', 'r') as f:
    CelebA_Gender_19_res = json.load(f)
path = '../Main[20]_Target[0]_29.json'
with open(f'./{path}', 'r') as f:
    CelebA_Gender_29_res = json.load(f)
path = '../Main[20]_Target[0]_59.json'
with open(f'./{path}', 'r') as f:
    CelebA_Gender_59_res = json.load(f)
path = '../IC_199.json'
with open(f'./{path}', 'r') as f:
    CelebA_IC_199_res = json.load(f)

methods_data = [
    ImageNet_res,
    # CelebA_Gender_19_res,
    CelebA_Gender_29_res,
    # CelebA_Gender_59_res,
    CelebA_IC_199_res
]
methods_name = [
    'ImageNet',
    # 'CelebA_Gender_19',
    'CelebA_Gender_29',
    # 'CelebA_Gender_59',
    'CelebA_IC_199'
]

indicators = list(CelebA_IC_199_res.keys())
num_indicators = len(indicators)
print("Num of attrs", num_indicators)

performance_data = []
variance_data = []
for data in methods_data:
    performance_data.append([data[key][0] for key in indicators])
    variance_data.append([data[key][1] for key in indicators])
performance_data = np.array(performance_data)
variance_data = np.array(variance_data)
num_rings = 6

# 随机生成示例数据（性能和方差）
performance_data = np.hstack((performance_data, performance_data[:, 0].reshape(-1,1)))
variance_data = np.hstack((variance_data, variance_data[:, 0].reshape(-1,1)))

# 设置角度
angles = np.linspace(0, 2 * np.pi, num_indicators, endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

# 设置圈数
radii = np.linspace(60, 100, num_rings)

# 设置要标注数值的角度
angle_to_label = np.pi / 4  # 这里设置为45度，也就是PI/4

# 创建雷达图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 绘制雷达图和方差
for i in range(len(methods_data)):
    ax.plot(angles, performance_data[i], label=f'{methods_name[i]}')
    # for j in range(num_indicators):
    #     ax.fill_between([angles[j], angles[(j + 1) % num_indicators]], performance_data[i, j] - variance_data[i, j],
    #                     performance_data[i, j] + variance_data[i, j], alpha=0.1)

# 添加数值标注
# for i in range(num_rings):
#     radius = radii[i]
#     value_to_label = 50 + i * 10
#     plt.text(angle_to_label, radius, str(value_to_label), ha='center', va='center', fontsize=8)

# 设置其他属性
ax.set_ylim(55, 95)  # 设置y轴范围，以便从50到100
ax.set_yticks([60, 70, 80, 90, 95])  # 设置刻度位置
ax.set_yticklabels([60, 70, 80, 90, 95])  # 设置刻度标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels([f'{PID_2_NAME[int(item)]}' for item in indicators], fontsize=8)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.savefig('./comparison.pdf', bbox_inches='tight')
# 显示图形
plt.show()

