
import pandas as pd
import matplotlib.pyplot as plt

# 数据
data = {
    'matrix_size': ['64x64', '128x128', '256x256', '512x512', '1024x1024'],
    'cpu': [0.000158, 0.001588, 0.013893, 0.122866, 2.057214],
    'proc:1': [0.000621, 0.0007, 0.002219, 0.007155, 0.079173],
    'proc:2': [0.000454, 0.000977, 0.001788, 0.005013, 0.400631],
    'proc:4': [0.000207, 0.000915, 0.001624, 0.005229, 0.316471],
    'proc:8': [0.000388, 0.00082, 0.001222, 0.003584, 0.287319]
}

# 转换为 DataFrame
df = pd.DataFrame(data)

# 绘制图表
plt.figure(figsize=(10, 6))

# 遍历不同的 matrix_size
for index, row in df.iterrows():
    plt.plot(['proc:1', 'proc:2', 'proc:4', 'proc:8'], row[['proc:1', 'proc:2', 'proc:4', 'proc:8']], label=f"Matrix Size {row['matrix_size']}")

plt.yscale('log')  # 设置y轴为对数刻度
plt.xlabel('Number of Processes')
plt.ylabel('Execution Time (s)')
plt.title('Matrix Multiplication Execution Time vs Number of Processes (Logarithmic Scale)')
plt.legend()
plt.grid(True)
plt.show()