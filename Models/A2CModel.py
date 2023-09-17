import torch
from torch import nn



class A2CModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 这里定义了一个body，这部分网络是Actor和Critic共用的
        self.body = nn.Sequential(
            nn.Linear(11, 64), nn.ReLU(inplace=True),  # 输入形状：(B, 11)，输出形状：(B, 64)
            nn.Linear(64, 128), nn.ReLU(inplace=True),  # 输入形状：(B, 64)，输出形状：(B, 128)
            nn.Linear(128, 256), nn.ReLU(inplace=True),  # 输入形状：(B, 128)，输出形状：(B, 256)
        )

        # 这个分支是critic单独享用的
        self.value_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True),  # 输入形状：(B, 256)，输出形状：(B, 128)
            nn.Linear(128, 64), nn.ReLU(inplace=True),   # 输入形状：(B, 128)，输出形状：(B, 64)
            nn.Linear(64, 1),   # 输入形状：(B, 64)，输出形状：(B, 1)
        )

        # 这个分支是actor单独享用的
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True),   # 输入形状：(B, 256)，输出形状：(B, 128)
            nn.Linear(128, 64), nn.ReLU(inplace=True),   # 输入形状：(B, 128)，输出形状：(B, 64)
            nn.Linear(64, 24),   # 输入形状：(B, 64)，输出形状：(B, 24)
        )

        # 额外的线性层用于输出阈值
        self.threshold_head = nn.Sequential(
            nn.Linear(256, 1)    # 输入形状：(B, 256)，输出形状：(B, 1)
        )

    def forward(self, x):
        x = self.body(x)   # 输入形状： (B,11)，输出形状： (B*256,)
        value = self.value_head(x)   # 输入形状： (B*256,)，输出形状： (B*1,)

        policy = self.policy_head(x)   # 输入形状： (B*256,)，输出形状： (B*24,)
        means = policy[:, :12]    # 取前12个值作为平均值
        stds = torch.exp(policy[:, 12:])    # 取后12个值作为标准差，并通过exp函数确保它们是正数

        threshold = self.threshold_head(x)   # 输入形状： (B*256,)，输出形状： (B*1,)

        return means, stds, value, threshold

    def getPolicyParams(self):
        return self.policy_head.parameters()

    def getValueParams(self):
        return [{'params': self.body.parameters()}, {'params': self.value_head.parameters()}]


if __name__ == '__main__':
    model = A2CModel()
    inp = torch.randn(4, 11)
    out = model(inp)
    print(out[0].shape, out[1].shape)
