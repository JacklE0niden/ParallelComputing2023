import pandas as pd
import matplotlib.pyplot as plt

# 数据
data = {
    'matrix_size': ['64x64', '128x128', '256x256', '512x512', '1024x1024'],
    'cpu': [0.000173, 0.001511, 0.014307, 0.129245, 2.048039],
    'proc:1': [0.000562, 0.000882, 0.002236, 0.015759, 0.134564],
    'proc:2': [0.000511, 0.000825, 0.002209, 0.014897, 0.115314],
    'proc:4': [0.000358, 0.000661, 0.00239, 0.014409, 0.106719],
}
# 转换为 DataFrame
df = pd.DataFrame(data)

# 绘制图表
plt.figure(figsize=(10, 6))

# 遍历不同的 matrix_size
for index, row in df.iterrows():
    plt.plot(['proc:1', 'proc:2', 'proc:4'], row[['proc:1', 'proc:2', 'proc:4']], label=f"Matrix Size {row['matrix_size']}")

plt.yscale('log')  # 设置y轴为对数刻度
plt.xlabel('Number of Processes')
plt.ylabel('Execution Time (s)')
plt.title('Matrix Multiplication Execution Time vs Number of Processes (Logarithmic Scale)')
plt.legend()
plt.grid(True)
plt.show()