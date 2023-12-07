import json

import matplotlib.pyplot as plt
import numpy as np

# with open(f'../vl/2023-11-14-23-50-03/log.json') as f:
#     log = json.load(f)
with open(f'../vl/2023-11-16-16-51-18/log.json') as f:
    log = json.load(f)

# def get_prop_acc_list(prop_id=0):
#     acc_list = []
#     for i in log.keys():
#         acc_list.append(log[i][prop_id])
#     return acc_list
#
#
# # 生成六组 y 值，分别代表六条曲线
# y1 = get_prop_acc_list(0)
# y2 = get_prop_acc_list(1)
# y3 = get_prop_acc_list(2)
# y4 = get_prop_acc_list(3)
# y5 = get_prop_acc_list(4)
# y6 = get_prop_acc_list(5)
#
# x = np.array(range(len(y6)))
# # 创建图表和子图
# plt.figure(figsize=(8, 6))
#
# # 绘制六条曲线
# plt.plot(x, y1, label='P1')
# plt.plot(x, y2, label='P2')
# plt.plot(x, y3, label='P3')
# plt.plot(x, y4, label='P4')
# plt.plot(x, y5, label='P5')
# plt.plot(x, y6, label='Main')
#
# # 添加标题和图例
# plt.title('The main and five auxiliary tasks.')
# plt.legend()
#
# # 显示图表
# plt.show()
train_lst = []
test_lst = []
for round_id, summary in log.items():
    train_lst.append(summary['TrainAcc'])
    test_lst.append(summary['TestAcc'])
plt.figure(figsize=(8, 6))
x = np.array(range(len(train_lst)))
plt.plot(x, train_lst, label='Train Acc.')
plt.plot(x, test_lst, label='Test Acc.')
plt.show()
