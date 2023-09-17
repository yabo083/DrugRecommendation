from EnvUtils import *


class Action:
    drugs = ["二甲双胍", "阿卡波糖", "格列美脲", "格列齐特", "瑞格列奈", "吡格列酮", "利格列汀", "沙格列汀", "利拉鲁肽",
             "艾塞那肽", "达格列净", "胰岛素"]

    form = ("药物剂量(二甲双胍={:.2f}mg, 阿卡波糖={:.2f}mg, 格列美脲={:.2f}mg, 格列齐特={:.2f}mg, 瑞格列奈={:.2f}mg, "
            "吡格列酮={:.2f}mg, 利格列汀={:.2f}mg, 沙格列汀={:.2f}mg, 利拉鲁肽={:.2f}mg, 艾塞那肽={:.2f}mg, "
            "达格列净={:.2f}mg, 胰岛素={:.2f}IU), 阈值={:.2f})")

    def __init__(self, doses: np.ndarray, threshold: float) -> None:
        self.doses = doses
        self.threshold = threshold

    def __str__(self) -> str:
        result = self.form.format(self.doses[0], self.doses[1], self.doses[2], self.doses[3], self.doses[4],
                                  self.doses[5], self.doses[6], self.doses[7], self.doses[8], self.doses[9],
                                  self.doses[10], self.doses[11], self.threshold)
        # result = (list(self.doses) + [self.threshold]).__str__()
        return result

    def __repr__(self) -> str:
        return self.__str__()

    def __format__(self, __format_spec: str) -> str:
        form = "{:" + __format_spec + "}"
        return form.format(self.__str__())

    def render(self):
        fig, ax = plt.subplots()
        ax.bar(self.drugs, self.doses)
        ax.axhline(y=self.threshold, color='r', linestyle='--')
        ax.set_xlabel('药物')
        ax.set_ylabel('剂量')
        ax.set_title('药物剂量分布')
        plt.xticks(rotation=45)
        plt.show()

    @staticmethod
    def generate_combination():
        """
        生成一个随机动作
        :return: 是一个定制好的药物剂量组合
        """
        drug = np.random.uniform(0.1, 1.0, len(Action.drugs))
        threshold = 0.5
        action = Action(drug, threshold)
        return action

    @staticmethod
    def makeBatch(actions: List["Action"]) -> torch.Tensor:
        doses = [action.doses for action in actions]
        thresholds = [action.threshold for action in actions]

        doses_tensor = torch.tensor(doses)
        thresholds_tensor = torch.tensor(thresholds).unsqueeze(1)

        batch = torch.cat((doses_tensor, thresholds_tensor), dim=1)

        return batch


if __name__ == '__main__':
    # 测试Action类
    action = Action.generate_combination()
    print(action)
