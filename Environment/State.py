from EnvUtils import *


class State:
    form = ("状态(血糖={:.2f}mmol/L, 血压={:.2f}mmHg, 体重={:.2f}℃, HbA1c={:.2f}%, GA={:.2f}%, 胰岛素={:.2f}μU/ml, C肽={:.2f}μg/L, "
            "酮体={:.2f}mmol/L, 乳酸/丙酮酸={:.2f}, 总胆固醇={:.2f}mmol/L, 尿微量白蛋白/肌酐={:.2f}mg/g)")

    def __init__(self, data: np.ndarray, tensor: torch.Tensor = None) -> None:
        self.data = data
        self.tensor = torch.from_numpy(data)

    def __str__(self) -> str:
        return self.form.format(*self.data.tolist())

    def render(self):
        """
        可视化状态，显示各项健康指标
        :return: 一个图表
        """
        labels = ['血糖', '血压', '体重', 'HbA1c', 'GA', '胰岛素', 'C肽', '酮体', '乳酸/丙酮酸', '总胆固醇',
                  '尿微量白蛋白/肌酐']
        values = self.data
        plt.figure(figsize=(10, 6))
        plt.barh(labels, values, color='skyblue')
        plt.xlabel('数值')
        plt.title('健康指标')
        plt.show()

    @staticmethod
    def generate_state():
        """
        生成一个随机状态
        :return: 一个随机状态的State对象
        """
        blood_sugar = np.random.uniform(3.9, 6.1)  # 血糖
        blood_pressure = np.random.uniform(90, 120)  # 血压
        body_weight = np.random.uniform(50, 100)  # 体重
        HbA1c = np.random.uniform(4, 6)  # 糖化血红蛋白
        GA = np.random.uniform(11, 17)  # 糖化白蛋白
        insulin = np.random.uniform(5, 25)  # 胰岛素
        C_peptide = np.random.uniform(0.8, 4.0)  # C肽
        ketone_body = np.random.uniform(0, 3)  # 酮体
        lactic_acid_acetoacetic_acid = np.random.uniform(0.5, 1.5)  # 乳酸/丙酮酸
        total_cholesterol = np.random.uniform(2.8, 5.2)  # 总胆固醇
        urinary_microalbumin_creatinine = np.random.uniform(0, 30)  # 尿微量白蛋白/肌酐

        data = np.array([blood_sugar, blood_pressure, body_weight, HbA1c, GA,
                         insulin, C_peptide, ketone_body, lactic_acid_acetoacetic_acid,
                         total_cholesterol,
                         urinary_microalbumin_creatinine])

        return State(data)

    @staticmethod
    def make_batch(states: List["State"]) -> torch.Tensor:
        """
        将状态列表转换为batch，神经网络的输入需要接收形如(B, C, H, W)的张量，其中：
        B是batch size，C是channel，H是height，W是width
        :param states: 状态列表
        :return: 状态的batch
        """
        return torch.stack([state.tensor for state in states])


if __name__ == '__main__':
    # 创建一个State对象并显示其健康指标
    state = State.generate_state()
    print(state)
