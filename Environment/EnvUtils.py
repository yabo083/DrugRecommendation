# Environment/EnvUtils.py
# Q: 为什么要新建这个文件？
# A: 为了引入一些库，定义一些游戏环境，State和Action都要用的值，可能也需要写一些方便的小函数

import numpy as np     # numpy库是非常强大的数学与数据处理库，我们处理矩阵和张量需要用到
# torch是PyTorch库，是很有名的深度学习库，它也能处理矩阵和张量，它的很多操作支持CUDA加速
# CUDA是配套Nvidia显卡的工具，能够最大化利用Nvidia显卡的优势加速大量的计算，深度学习最好使用显卡加速
import torch
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# typing是python的类型注释库，它可以帮助你给函数以及变量标注类型，使得你的项目更加清晰和可维护
# 对于被标注了类型的变量，PyCharm可以更好地提供补全建议，因为它可以找到变量对应的类的定义
from typing import *


# 这里定义一个类型别名叫RowCol，我将会把它用作一个包含行和列的2维向量，这里其实就是把numpy array另起了个名字
RowCol = np.ndarray
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

# 这里定义设备，PyTorch可以自行决定是要用显卡处理数据，还是用cpu处理
# 这取决于你电脑硬件的配置，以及安装的PyTorch版本是否支持CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

