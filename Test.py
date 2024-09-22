import torch
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, roc_auc_score

class Test:
    def __init__(self, save_path) -> None:
        """
            self._y_probs: 存放所有测试数据的正类(positive)概率
        """
        self._y_probs = []
        self._y_trues = []
        self.path = save_path  # 保存路径
    
    # add test data
    def append_data(self, y_prob, y_true) -> None:
        self._y_probs.append(y_prob)
        self._y_trues.append(y_true)

    def draw_roc(self, filename, dpi=300):
        """
        Draw ROC
        """
        y_probs = torch.cat(self._y_probs).numpy()
        y_trues = torch.cat(self._y_trues).numpy()
        # 计算 FPR, TPR 和阈值
        fpr, tpr, thresholds = roc_curve(y_trues, y_probs)

        # 计算 AUC 分数
        auc = roc_auc_score(y_trues, y_probs)

        # 绘制 ROC 曲线
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 绘制对角线
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(self.path + filename, dpi=dpi)  # dpi 控制图像的分辨率
        plt.close()  # 可选：关闭当前图像，释放内存
    
    def draw_pr(self, filename,dpi=300):
        """
        Draw Precision-Recall Curve
        """
        y_probs = torch.cat(self._y_probs).numpy()
        y_trues = torch.cat(self._y_trues).numpy()
        # 计算 Precision, Recall 和阈值
        precision, recall, thresholds = precision_recall_curve(y_trues, y_probs)

        # 计算 AP 分数
        average_precision  = average_precision_score(y_trues, y_probs)

        # 绘制 PR 曲线
        # 绘制 PR 曲线
        plt.figure()
        plt.plot(recall, precision, color='b', lw=2, label=f'PR curve (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        #plt.show()
        plt.savefig(self.path + filename, dpi=dpi)  # dpi 控制图像的分辨率
        plt.close()  # 可选：关闭当前图像，释放内存

if __name__ == "__main__":

    t = Test('./')
    y_probs_iter_1 = torch.tensor([0.1, 0.4, 0.35, 0.8])
    y_probs_iter_2 = torch.tensor([0.2, 0.5, 0.30, 0.7])
    y_probs_iter_3 = torch.tensor([0.15, 0.45, 0.25, 0.75])
    y_probs_iter_4 = torch.tensor([0.12, 0.48, 0.28, 0.78])
    y_probs_iter_5 = torch.tensor([0.13, 0.46, 0.27, 0.77])
    # 对应的真实标签
    y_true_iter_1 = torch.tensor([0, 1, 0, 1])
    y_true_iter_2 = torch.tensor([0, 1, 0, 1])
    y_true_iter_3 = torch.tensor([0, 1, 0, 1])
    y_true_iter_4 = torch.tensor([0, 1, 0, 1])
    y_true_iter_5 = torch.tensor([0, 1, 0, 1])

    t.append_data(y_probs_iter_1, y_true_iter_1)
    t.append_data(y_probs_iter_2, y_true_iter_2)
    t.append_data(y_probs_iter_3, y_true_iter_3)
    t.append_data(y_probs_iter_4, y_true_iter_4)
    t.append_data(y_probs_iter_5, y_true_iter_5)

    t.draw_roc('roc.png')
    t.draw_pr('pr.png')