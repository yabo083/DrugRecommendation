from typing import Tuple, Any

from Agents.AgentUtils import *
from Models.A2CModel import A2CModel
from Models.ModelUtils import loadModel, saveModel, copyModel
import torch.nn.functional as F


class A2CAgent:
    # 像前几小节一样，我们假设这个类需要一个字典作为输入，东西已经定义好放在字典里了
    # 在之后我们可以写yaml配置文件把数值都定义好。
    def __init__(self, agent_cfg: Dict[str, Any]):
        # 先定义policy model，这个模型是我们真正一直在训练的模型
        # 输出policy, V(s)和getAction中都要使用policy model
        self.policy_model = A2CModel()

        # 这里加载模型，因为你不一定一次性把模型训练好，在训练时可能暂停
        # 暂停后重启训练程序总不能从头开始训练吧，这时就要加载已经存储到本地的模型参数，继续训练
        if agent_cfg['load_model']:
            print(f"Loading model from {agent_cfg['load_model']}")
            loadModel(self.policy_model, agent_cfg['load_model'])
        # 将模型转移到cuda上
        self.policy_model.to(device)

        # 然后是target model，它只用来输出V(s')，也就是下一个state的value
        # 先让target model与policy model完全一致，在之后每隔一段时间，我们就会更新一次target model
        self.target_model = A2CModel()
        copyModel(self.policy_model, self.target_model)
        self.target_model.to(device)

        # policy model需要被训练和更新参数，因此设为train()模式，target model不需要，因此设为eval()模式
        self.policy_model.train()
        self.target_model.eval()

        # 此处定义优化器，使用Adam优化器，因为它简单有效
        # 注意我们使用了两个优化器，分别优化模型定义中policy head的参数，以及value head+body的参数
        # policy head的参数只有value head的参数的1/10
        self.optimizer_value = torch.optim.Adam(self.policy_model.getValueParams(), lr=agent_cfg['value_lr'])
        self.optimizer_policy = torch.optim.Adam(self.policy_model.getPolicyParams(), lr=agent_cfg['policy_lr'])

        # 定义损失函数，这个只为了求value loss (critic loss)
        self.loss_func = nn.MSELoss()

        # epsilon即模型在getAction时，采取随机行动的概率
        self.epsilon = agent_cfg['epsilon']
        # gamma，折扣因子
        self.gamma = agent_cfg['gamma']

    def update(self, batch_data: TensorTuple) -> tuple[Any, Any, Any, Any]:
        """
        A2C算法的更新过程
        :param batch_data: [states, actions, rewards, next_states, terminate_flags]
        :return:
        """
        states, actions, rewards, next_states, terminate_flags = batch_data
        # 别忘了要时刻能说出每个数据的尺寸
        # states: (B, 11)
        # actions: (B, 13)
        # rewards: (B,)
        # next_states: (B, 11)
        # terminate_flags: (B,)

        # 这里可以在DEBUG时增加一个断点，用来判断：
        # 1. 各个输入数据的尺寸是否正确？比如action的尺寸时(B, 1)还是(B, )。
        # 2. 各个输入数据的数值是否合理？比如states中是否偶然存在某些数值超过1000？

        # TD_target = r + gamma * V(s')
        # 这里的with torch.no_grad()表示在这个代码块里的运算是不需要保存梯度的
        # 因为这部分内容不参与梯度下降与参数更新
        with torch.no_grad():
            _, _, V_s_next, _ = self.target_model(next_states)
            td_target = rewards + self.gamma * V_s_next * (1 - terminate_flags)

        means, stds, V_s, threshold = self.policy_model(states)
        # 创建一个正态分布
        policy = torch.distributions.Normal(means, stds)
        entropy = policy.entropy().mean()

        advantage = (td_target - V_s).detach()
        # 计算log_prob
        log_prob = policy.log_prob(actions[:, :12]).sum(dim=1)
        policy_loss = (-log_prob * advantage - 0.01 * entropy).mean()

        value_loss = self.loss_func(V_s, td_target)

        # 添加阈值损失
        threshold_loss = self.loss_func(threshold, actions[:, 12])

        total_loss = policy_loss + value_loss + threshold_loss

        # 这里执行参数更新，zero_grad()是清除旧的梯度，因为PyTorch的参数梯度可以叠加，但我们目前不希望它叠加
        self.optimizer_policy.zero_grad()
        self.optimizer_value.zero_grad()
        # 反向传播计算梯度
        total_loss.backward()
        # 参数更新
        self.optimizer_policy.step()
        self.optimizer_value.step()

        # 返回三个loss的值，用来打印和可视化
        return policy_loss.item(), value_loss.item(), threshold_loss.item(), total_loss.item()

    def updateTargetModel(self):
        """ 这里是将policy model全盘拷贝到target model，RL算法中时不时就要这么做 """
        copyModel(self.policy_model, self.target_model)

    # @torch.no_grad()可以表明这个方法里的操作都是不求梯度的
    @torch.no_grad()
    def getAction(self, state: State) -> Action:
        # 有epsilon的概率取随机行动
        if random.random() < self.epsilon:
            return Action.randomAction()
        else:
            # 有1-epsilon的概率取最好的行动
            policy, _, _, threshold = self.policy_model(state.tensor.unsqueeze(0))
            return Action(policy.argmax(dim=1).item(), threshold.item())

    def train(self):
        """ 设置当前神经网络为训练模式 """
        self.policy_model.train()

    def eval(self):
        """ 设置当前神经网络为非训练模式 """
        self.policy_model.eval()

    def save(self, model_path: str):
        """ 保存当前的神经网络 """
        saveModel(self.policy_model, model_path)
