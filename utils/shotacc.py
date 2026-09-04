import random
from PIL import ImageFilter
import numpy as np
import torch


def shot_acc(preds, labels, cls_num_list, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    """
    更稳健的长尾精度评测脚本。
    :param preds: 预测的标签 (numpy array 或 tensor)
    :param labels: 真实的标签 (numpy array 或 tensor)
    :param cls_num_list: 每个类别的训练集样本数列表 (e.g., [1280, 1000, ..., 5])
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
    # 确保 cls_num_list 是常规的 list 或 array
    if isinstance(cls_num_list, np.ndarray):
        cls_num_list = cls_num_list.tolist()

    num_classes = len(cls_num_list)
    class_correct = np.zeros(num_classes)
    test_class_count = np.zeros(num_classes)
    
    # 统计每个类别的预测对的数量和总数量
    for l in range(num_classes):
        class_mask = (labels == l)
        test_class_count[l] = np.sum(class_mask)
        class_correct[l] = np.sum(preds[class_mask] == labels[class_mask])
        
    many_shot = []
    median_shot =[]
    low_shot =[]
    
    # 根据 cls_num_list 进行分段
    for i in range(num_classes):
        if test_class_count[i] == 0:
            continue  # 防止验证集中某个类别缺失导致除以0
            
        acc = class_correct[i] / test_class_count[i]
        
        if cls_num_list[i] > many_shot_thr:
            many_shot.append(acc)
        elif cls_num_list[i] < low_shot_thr:
            low_shot.append(acc)
        else:
            median_shot.append(acc)
            
    # 计算均值，如果列表为空则返回 0
    many_acc = np.mean(many_shot) if len(many_shot) > 0 else 0.0
    med_acc = np.mean(median_shot) if len(median_shot) > 0 else 0.0
    few_acc = np.mean(low_shot) if len(low_shot) > 0 else 0.0
    
    # 统一乘以 100，使其与 Top-1 Acc 的百分比单位保持一致 (例如 65.4)
    many_acc *= 100.0
    med_acc *= 100.0
    few_acc *= 100.0
    
    if acc_per_cls:
        class_accs = [class_correct[i] / test_class_count[i] * 100.0 if test_class_count[i] > 0 else 0 
                      for i in range(num_classes)]
        return many_acc, med_acc, few_acc, class_accs
    else:
        return many_acc, med_acc, few_acc



