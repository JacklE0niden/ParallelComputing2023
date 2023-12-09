import matplotlib.pyplot as plt

# 数据
matrix_sizes = ['64x64', '128x128', '256x256', '512x512', '1024x1024']
proc_1 = [1.042671614, 1.046040516, 0.826555024, 1.02073882, 0.984441062]
proc_2 = [4.074534161, 3.813953488, 1.433436533, 2.496, 2.643515471]
proc_4 = [14.03225806, 3.602649007, 2.614864865, 5.30964467, 6.128101558]

# 绘制折线图
plt.plot(matrix_sizes, proc_1, marker='o', label='proc:1')
plt.plot(matrix_sizes, proc_2, marker='o', label='proc:2')
plt.plot(matrix_sizes, proc_4, marker='o', label='proc:4')

# 添加标签和标题
plt.xlabel('Matrix Size')
plt.ylabel('Time')
plt.title('Matrix Multiplication Time Comparison ( Col time / Row time )')
plt.legend()

# 显示图形
plt.show()