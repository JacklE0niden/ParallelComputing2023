# 并行分布式计算期末程序设计项目代码库

- OpenMP基于C++的<omp.h>库，编译链接时需要添加-lm -fopenmp
- MPI基于C++的<mpi.h>库，编译需要使用mpicxx，并且运行时需要使用特殊的指令格式（详见examples）

- 文件夹说明
  - build 存储编译好的可执行文件
  - examples 一些shell可执行脚本的实例
  - output 输出结果的实例
  - plot_data excels 中有测试好的性能表格 根据性能表格绘制的对比图和绘图的python文件
  - src c++源代码（可编译运行）

- 运行说明
  - 在Ubuntu(GNU/Linux wsl)环境下执行make
  - 运行 ./test_correct.sh 可以测试程序的正确性(10个线程，1000的数组大小)
  - examples 目录下有shell可执行脚本的实例，可以解注释来执行
  - 也可以在 build 目录下(有编译好的可执行文件)，直接通过相关命令执行