import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

# 读取 Excel 文件
file_path = os.path.join(script_dir, './excels/PQS_OpenMP_O2_speedup.xlsx')
df = pd.read_excel(file_path)

# 绘制图表
fig, ax = plt.subplots()
for col in df.columns[1:]:
    ax.plot(df['n/thread'], df[col], marker='o', label=f'n={col}')
  

ax.set_xlabel('Number of Threads')
ax.set_ylabel('SpeedUp Ratio')
ax.set_title('SpeedUp Ratio vs. Number of Threads')
ax.legend(title='Data Size (n)')

plt.show()       
