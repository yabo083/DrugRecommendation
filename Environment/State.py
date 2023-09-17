import numpy as np

from EnvUtils import *


class State:
    form = ("状态(血糖={:.2f}mmol/L, 血压={:.2f}mmHg, 体重={:.2f}℃, HbA1c={:.2f}%, GA={:.2f}%, 胰岛素={:.2f}μU/ml, "
            "C肽={:.2f}μg/L, "
            "酮体={:.2f}mmol/L, 乳酸/丙酮酸={:.2f}, 总胆固醇={:.2f}mmol/L, 尿微量白蛋白/肌酐={:.2f}mg/g)")

    # 构造初始化函数
    def __init__(self, tensor: np.ndarray) -> None:
        self.tensor = torch.Tensor(tensor)   # 直接将传入的张量数组赋值给tensor属性
        self.data = tensor  # 直接将传入的张量数组赋值给data属性
        # 将data中的数据变成list
        self.blood_sugar = self.data[0]
        self.blood_pressure = self.data[1]
        self.body_weight = self.data[2]
        self.HbA1c = self.data[3]
        self.GA = self.data[4]
        self.insulin = self.data[5]
        self.C_peptide = self.data[6]
        self.ketone_body = self.data[7]
        self.lactic_acid_acetoacetic_acid = self.data[8]
        self.total_cholesterol = self.data[9]
        self.urinary_microalbumin_creatinine = self.data[10]


    def __str__(self) -> str:
        return self.form.format(self.blood_sugar, self.blood_pressure, self.body_weight, self.HbA1c,
                                self.GA, self.insulin, self.C_peptide, self.ketone_body,
                                self.lactic_acid_acetoacetic_acid, self.total_cholesterol,
                                self.urinary_microalbumin_creatinine)

    def render(self):
        """
        可视化状态，显示各项健康指标
        :return: 一个图表
        """
        labels = ['血糖', '血压', '体重', 'HbA1c', 'GA', '胰岛素', 'C肽', '酮体', '乳酸/丙酮酸', '总胆固醇',
                  '尿微量白蛋白/肌酐']
        values = [
            self.blood_sugar, self.blood_pressure, self.body_weight, self.HbA1c,
            self.GA, self.insulin, self.C_peptide, self.ketone_body,
            self.lactic_acid_acetoacetic_acid, self.total_cholesterol,
            self.urinary_microalbumin_creatinine
        ]
        plt.figure(figsize=(10, 6))
        plt.barh(labels, values, color='skyblue')
        plt.xlabel('数值')
        plt.title('健康指标')
        plt.show()

    @staticmethod
    def generate_state() -> "State":
        """
        生成一个随机状态，并进行标准化处理
        :return: 一个随机状态的标准化State对象
        """
        blood_sugar = np.random.uniform(3.9, 6.1)
        blood_pressure = np.random.uniform(90, 120)
        body_weight = np.random.uniform(50, 100)
        HbA1c = np.random.uniform(4, 6)
        GA = np.random.uniform(11, 17)
        insulin = np.random.uniform(5, 25)
        C_peptide = np.random.uniform(0.8, 4.0)
        ketone_body = np.random.uniform(0, 3)
        lactic_acid_acetoacetic_acid = np.random.uniform(0.5, 1.5)
        total_cholesterol = np.random.uniform(2.8, 5.2)
        urinary_microalbumin_creatinine = np.random.uniform(0, 30)

        state = State(np.array([blood_sugar, blood_pressure, body_weight, HbA1c,
                                GA, insulin, C_peptide, ketone_body,
                                lactic_acid_acetoacetic_acid, total_cholesterol,
                                urinary_microalbumin_creatinine]))

        # 将生成的随机状态进行标准化
        normalized_state = State.normalize_state(state)

        return normalized_state

    @staticmethod
    def makeBatch(states: List["State"]) -> torch.Tensor:
        """
        将状态列表转换为batch，神经网络的输入需要接收形如(B, C, H, W)的张量，其中：
        B是batch size，C是channel，H是height，W是width
        :param states: 状态列表
        :return: 状态的batch(B, 11)
        """
        return torch.stack([torch.Tensor([
            state.blood_sugar, state.blood_pressure, state.body_weight, state.HbA1c,
            state.GA, state.insulin, state.C_peptide, state.ketone_body,
            state.lactic_acid_acetoacetic_acid, state.total_cholesterol,
            state.urinary_microalbumin_creatinine
        ]) for state in states])

    @staticmethod
    def normalize_state(state: "State") -> "State":
        """
        将State对象中的数据进行标准化处理, 使得数据的范围在[-1, 1]之间, 但这只是为了让代码跑起来，实际上这样做是不合理的
        :param state: 要标准化的State对象
        :return: 标准化后的State对象
        """
        data = state.data

        # 定义标准化的范围和均值
        ranges = [2.2, 30.0, 50.0, 2.0, 6.0, 20.0, 3.2, 1.5, 1.0, 2.4, 30.0]
        means = [5.0, 105.0, 75.0, 5.0, 14.0, 12.5, 2.4, 0.8, 1.1, 4.0, 10.0]

        # 标准化数据
        normalized_data = (data - np.array(means)) / np.array(ranges)

        # 创建新的标准化State对象并返回
        return State(normalized_data)


if __name__ == '__main__':
    # 创建一个State对象并显示其健康指标
    state = State.generate_state()
    print(state)
