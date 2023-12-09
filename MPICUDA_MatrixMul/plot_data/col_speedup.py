import matplotlib.pyplot as plt
import numpy as np

# 数据
matrix_sizes = ['64x64', '128x128', '256x256', '512x512', '1024x1024']
cpu_time = [1, 1, 1, 1, 1]
speedup_1 = [0.307829181,2.660211268,20.70477569,82.06031746,351.8362824]
speedup_2 = [0.263719512,2.303353659,15.45032397,69.04113248,255.0166853]
speedup_4 = [0.198850575,2.777573529,18.48449612,41.18706182,192.8473635]

# 计算加速比相对于单处理器
speedup_1_rel = np.array(speedup_1) / np.array(cpu_time)
speedup_2_rel = np.array(speedup_2) / np.array(cpu_time)
speedup_4_rel = np.array(speedup_4) / np.array(cpu_time)

# 绘制柱状图
plt.figure(figsize=(12, 6))
bar_width = 0.2
index = np.arange(len(matrix_sizes))

plt.bar(index, speedup_1_rel, bar_width, label='1 Processors')
plt.bar(index + bar_width, speedup_2_rel, bar_width, label='2 Processors')
plt.bar(index + 2 * bar_width, speedup_4_rel, bar_width, label='4 Processors')

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

plt.xlabel('Matrix Size')
plt.ylabel('Relative Speedup')
plt.title('Relative Speedup vs Matrix Size and Processors')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()