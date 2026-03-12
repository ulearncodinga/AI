''''''

'''
首先求出 IOU=0.5时的数量
求解Precision 和 Recall

AP (Average Precision)
对于每个类别 AP是 0-1 的值 表示在不同阈值下Precision-Recall曲线下的面积

置信度 通常是分类器的输出概率

num_ob: 表示真实标注的数量
GT ID: 真实标注的目标ID
Confidence:模型预测的置信度的得分
OB(IoU=0.5):表示在IoU阈值为0.5时,预测框是否被认为是有效的



实际应用时,让confidence保持一个阈值

Precision = TP / (TP+FP)

Recall = TP / (TP + FN)



非极大值抑制:
NMS 后处理机制 常用于目标检测任务中
以减少荣誉的检测结果 保留最有可能正确的预测框
主要目的是从多个重叠的候选框中选择出最佳的一个,避免预测框对应同一个目标








'''