# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# script_dir = os.path.dirname(os.path.abspath(__file__))

# # 读取 Excel 文件
# file_path = os.path.join(script_dir, './excels/PQS_OpenMP_speedup.xlsx')
# df = pd.read_excel(file_path)

# # 绘制图表
# fig, ax = plt.subplots()
# for col in df.columns[1:]:
#     ax.plot(df['n/thread'], df[col], marker='o', label=f'n={col}')
  

# ax.set_xlabel('Number of Threads')
# ax.set_ylabel('SpeedUp Ratio')
# ax.set_title('SpeedUp Ratio vs. Number of Threads')
# ax.legend(title='Data Size (n)')

# plt.show()       

import matplotlib.pyplot as plt
import numpy as np

# 数据
matrix_sizes = ['64x64', '128x128', '256x256', '512x512', '1024x1024', '2048x2048']
cpu_time = [1, 1, 1, 1, 1, 1]
speedup_1 = [0.293135436, 2.924493554, 16.61842105, 79.62799741, 347.9137494, 705.1573034]
speedup_2 = [0.98136646, 9.23255814, 21.50619195, 163.8213333, 677.160632, 778.7285779]
speedup_4 = [2.548387097, 10.51655629, 46.93581081, 207.8950931, 1187.082516, 1449.715063]
speedup_8 = [2.633333333, 21.75342466, 66.15714286, 361.3705882, 1758.302564, 4482.785714]

# 计算加速比相对于单处理器
speedup_1_rel = np.array(speedup_1) / np.array(cpu_time)
speedup_2_rel = np.array(speedup_2) / np.array(cpu_time)
speedup_4_rel = np.array(speedup_4) / np.array(cpu_time)
speedup_8_rel = np.array(speedup_8) / np.array(cpu_time)

# 绘制柱状图
plt.figure(figsize=(12, 6))
bar_width = 0.2
index = np.arange(len(matrix_sizes))

plt.bar(index, speedup_1_rel, bar_width, label='1 Processors')
plt.bar(index + bar_width, speedup_2_rel, bar_width, label='2 Processors')
plt.bar(index + 2 * bar_width, speedup_4_rel, bar_width, label='4 Processors')
plt.bar(index + 3 * bar_width, speedup_8_rel, bar_width, label='8 Processors')

plt.xlabel('Matrix Size')
plt.ylabel('Relative Speedup')
plt.title('Relative Speedup vs Matrix Size and Processors')
plt.xticks(index + bar_width, matrix_sizes)
plt.legend()
plt.tight_layout()

plt.show()

# 绘制曲线图
plt.figure(figsize=(12, 6))

plt.plot(matrix_sizes, speedup_1_rel, marker='o', label='1 Processors')
plt.plot(matrix_sizes, speedup_2_rel, marker='o', label='2 Processors')
plt.plot(matrix_sizes, speedup_4_rel, marker='o', label='4 Processors')
plt.plot(matrix_sizes, speedup_8_rel, marker='o', label='8 Processors')

plt.xlabel('Matrix Size')
plt.ylabel('Relative Speedup')
plt.title('Relative Speedup vs Matrix Size and Processors')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()