from Environment.AgentModel import AgentModel
from Environment.EnvUtils import *
from Environment.Action import Action
from Environment.State import State


class EnvController:
    form = "环境" + "-" * 62 + "+\n|  {}|\n|  {:<56}|\n|  {}|\n|  奖励({:.4f})" + " " * 51 + "|\n+" + "-" * 65 + "+"

    def __init__(self):
        """
        初始化环境控制器
        """
        self.agent_model = AgentModel()  # 创建代理模型实例

        # 初始化随机状态
        self._state = State(np.zeros((10, 10)))

        # 记录上一次的状态、动作、奖励
        self.action: Action = None
        self.state: State = None
        self.reward: float = None

        self.t: int = 0  # 记录游戏的时间
        self.game_over = False

    def reset(self):
        """
        重置环境
        :return:
        """
        self._state = State(np.zeros((10, 10)))
        self.game_over = False

    def step(self, action: Action) -> Tuple[State, float, bool]:
        """
        这个函数是实现环境动态的核心，也是RL算法和环境之间的接口
        :param action: 要执行的动作
        :return: 返回新的state，reward，和一个布尔值表示游戏是否结束
        """
        current_state = self._state.data
        # 输入当前状态和动作到代理模型中获取下一个状态
        next_state_data = self.agent_model.predict(current_state, action)

        # 更新状态
        self.action = action
        self.state = State(next_state_data)
        self.reward = self.calculate_reward()  # 计算奖励
        self.game_over = self._check_game_over()  # 检查游戏是否结束
        self.t += 1  # 更新游戏时间

        return self.state, self.reward, self.game_over

    def calculate_reward(self) -> float:
        """
        计算奖励
        :return: 返回奖励值
        """
        normal_range = np.array([
            [3.9, 6.1],  # 血糖
            [90, 120],  # 血压
            [50, 100],  # 体重
            [4, 6],  # 糖化血红蛋白
            [11, 17],  # 糖化白蛋白
            [5, 25],  # 胰岛素
            [0.8, 4.0],  # C肽
            [0, 3],  # 酮体
            [0.5, 1.5],  # 乳酸/丙酮酸
            [2.8, 5.2],  # 总胆固醇
            [0, 30]  # 尿微量白蛋白/肌酐
        ])  # 正常范围

        if isinstance(self.state.data, memoryview):
            # 转换为np.ndarray
            state_data = np.array(self.state.data)
        else:
            state_data = self.state.data

        # 计算normal_range的每个范围的平均值
        normal_mean = np.mean(normal_range, axis=1)

        mse_sum = np.sum((state_data - normal_mean) ** 2)  # 计算均方误差

        # 判断是否达到边界
        boundary_threshold = 0.1  # 边界阈值，可以根据实际情况调整
        boundary_penalty = 0.5  # 边界惩罚值，可以根据实际情况调整
        # 判断是否达到边界
        if np.any(np.abs(state_data - normal_mean) > boundary_threshold):
            reward = -mse_sum - boundary_penalty  # 处于边界时添加边界惩罚
        else:
            reward = -mse_sum  # 未达到边界时不添加边界惩罚

        return reward

    def _check_game_over(self) -> bool:
        # 检查游戏是否结束
        # 这里根据需要进行判断，例如达到一定的步数或者达到某个特定的状态等
        if self.t > 100:  # 示例中的游戏结束条件，可以根据实际情况修改
            return True
        else:
            return False

    def __str__(self):
        # 渲染游戏环境的可视化
        # 这里可以根据实际需求选择适合的方式进行可视化，例如打印状态数据、绘制图形等
        return self.form.format(self.state, self.action, self.state, self.reward)


if __name__ == '__main__':
    # 创建一个环境控制器
    env_controller = EnvController()

    env_controller.step(Action.generate_combination())

    print(env_controller)
