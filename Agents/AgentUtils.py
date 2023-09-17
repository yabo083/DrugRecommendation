from typing import *
import random
import torch
import torch.nn as nn
from Environment.State import State
from Environment.Action import Action

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义一个类型表示多个torch.Tensor
TensorTuple = Tuple[torch.Tensor, ...]


# 在上面的定义中，我们需要用到s, a, r, s'和is_terminate这样的5元组
# 在这里，我们就定义这个5元组为一个类，方便处理
class MemoryTuple:
    def __init__(self, state: State, action: Action, reward: float, next_state: State, terminate_flag: bool):
        # 这里只是单纯地存储5个值
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminate_flag = terminate_flag

    def __iter__(self):
        # 因为要使用zip，所以必须实现__iter__方法
        # 当你想要循环这一类的实例时，python会调用__iter__方法
        # yield语句的作用类似于return，但它会记住此函数的进度，下一次从断掉的地方继续执行
        # 每一次调用执行__iter__方法，都会返回一个不同的数据
        yield self.state
        yield self.action
        yield self.reward
        yield self.next_state
        yield self.terminate_flag

    @staticmethod
    def makeBatch(batch_data: List['MemoryTuple']) -> TensorTuple:
        """ 就像State和Action的makeBatch一样，这里我们也打包一系列5元组的batch数据 """
        # 首先，*batch_data可以将batch_data拆解开，变成B个独立的5元组
        # 然后，zip()会把所有这些独立的5元组中的每个元素单独拿出来合并在一起，这一步涉及到循环，也因此必须完成上面的__iter__方法
        # 最终变成5个长度为B的List，每个只包含一种数据
        states, actions, rewards, next_states, terminate_flags = zip(*batch_data)
        # 对于一系列的states，actions，next_states，只需要使用State和Action类定义好的makeBatch方法即可
        # 对于reward和terminate_flags，则要在这里直接变为tensor类型
        return State.makeBatch(states), \
            Action.makeBatch(actions), \
            torch.tensor(rewards, dtype=torch.float32, device=device), \
            State.makeBatch(next_states), \
            torch.tensor(terminate_flags, dtype=torch.float32, device=device)
