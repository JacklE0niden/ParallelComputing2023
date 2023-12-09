import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

# 读取 Excel 文件
file_path = os.path.join(script_dir, './excels//PQS_OpenMP.xlsx')
df = pd.read_excel(file_path)

# 绘制图表
fig, ax = plt.subplots()
for col in df.columns[1:]:
    ax.plot(df['n/thread'], df[col], marker='o', label=f'n={col}')



 
ax.set_yscale('log')  

ax.set_xlabel('Number of Threads')
ax.set_ylabel('Execution Time')
ax.set_title('Execution Time vs. Number of Threads')
ax.legend(title='Data Size (n)')

plt.show()       

# 绘制图表
fig, ax = plt.subplots()
for col in df.columns[1:]:
    ax.plot(df['n/thread'], df[col], marker='o', label=f'n={col}')

ax.set_xlabel('Number of Threads')
ax.set_ylabel('Execution Time')
ax.set_title('Execution Time vs. Number of Threads')
ax.legend(title='Data Size (n)')

plt.show()


ax.set_yscale('log')

ax.set_xlabel('Number of Threads')
ax.set_ylabel('Execution Time')
ax.set_title('Execution Time vs. Number of Threads')
ax.legend(title='Data Size (n)')

plt.show()

fig_3d = plt.figure()
ax_3d = fig_3d.add_subplot(111, projection='3d')

# 遍历数据，为每个数据点绘制散点图
for col in df.columns[1:]:
    ax_3d.scatter(df['n/thread'], [int(col)] * len(df), df[col], label=f'n={col}')

ax_3d.set_xlabel('Number of Threads')
ax_3d.set_ylabel('Data Size (n)')
ax_3d.set_zlabel('Execution Time')
ax_3d.set_title('Execution Time vs. Number of Threads and Data Size')

ax_3d.legend()

plt.show()