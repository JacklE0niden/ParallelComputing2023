import os
import pandas as pd
import matplotlib.pyplot as plt
# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 构造 Excel 文件的绝对路径
file_path = os.path.join(script_dir, './excels/test_serial.xlsx')

# # 读取 Excel 表格
# df = pd.read_excel(file_path)


# # 读取 Excel 表格
# file_path = '../test_serial.xlsx'  # 替换成你的 Excel 文件相对路径
df = pd.read_excel(file_path)

# # 提取数据
# data_sizes = df['n']
# runtimes = df['time']

# # 绘制图表
# plt.plot(data_sizes, runtimes, marker='o', linestyle='-', color='b')
# plt.xlabel('数据量')
# plt.ylabel('运行时间')
# plt.title('运行时间随数据量变化')
# plt.grid(True)
# plt.savefig('test_serial.png')  # 图片保存路径，替换成你想保存的位置
# plt.show()

# 绘制图表
plt.plot(df['n'], df['time'], marker='o', linestyle='-', color='b')
# plt.xscale('log')  # 使用对数刻度，因为数据范围较大
plt.xlabel('amount')
plt.ylabel('time')
plt.title('time vs amount of data')
plt.grid(True)
plt.savefig('test_serial.png')
plt.show()


plt.plot(df['n'], df['time'], marker='o', linestyle='-', color='b')
plt.xscale('log')  # 使用对数刻度，因为数据范围较大
plt.xlabel('amount(log scale)')
plt.ylabel('time')
plt.title('time vs amount of data')
plt.grid(True)
plt.savefig('test_serial(log).png')
plt.show()
print("ok")