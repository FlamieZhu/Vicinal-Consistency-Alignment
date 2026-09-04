import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class MixedCEKDLoss(nn.Module):
    def __init__(self, cls_num_list=None):
        super().__init__()
        self.cls_num_list = torch.tensor(cls_num_list)
        self.register_buffer('log_prior', torch.log(self.cls_num_list / self.cls_num_list.sum()))
        self.C_number = len(cls_num_list)  # class number
        self.base_loss = F.cross_entropy

    def forward(self, logits, logits_k, logits_m, logits_s, targets, args=None):

        ce_loss = 0
        sc_loss = 0
        ded_loss = 0

        batch_size = targets.shape[0]

        tea_ens_all = 0

        # Build the DED target from both EMA views and all experts.
        with torch.no_grad():
            for i in range(args.num_experts):
                tea_ens_all += logits_k[i] +  self.log_prior

            tea_ens_all_v1, tea_ens_all_v2 = (tea_ens_all).chunk(2, dim=0)
            tea_ens_all = (tea_ens_all_v1 + tea_ens_all_v2).detach() / (2 * args.num_experts)
            tea_label = torch.max(tea_ens_all, 1)[1]
            
            tea_prob_all = F.softmax(tea_ens_all, dim=1)
            


        for idx in range(args.num_experts):
            tea_v1, tea_v2 = (logits_k[idx] +  self.log_prior).chunk(2, dim=0)
            tea_prob = F.softmax(0.5*tea_v1 + 0.5*tea_v2, dim=1).detach()
            stu_log_prob_m = F.log_softmax((logits_m[idx] +  self.log_prior) / 1., dim=1)

            stu_label = torch.max((logits_s[idx] +  self.log_prior).detach(), 1)[1]
            stu_log_prob_s = F.log_softmax((logits_s[idx] +  self.log_prior) / 1., dim=1)

            # CKF drops cases where a correct student would learn from a wrong teacher.
            select_index = (tea_label != targets) & (stu_label == targets)
            select_index = ~select_index

            if torch.sum(select_index) > 0:
                ded_loss += F.kl_div(stu_log_prob_s[select_index], tea_prob_all[select_index], reduction='batchmean')

            # SC transfers each expert's EMA prediction to the mixed view.
            sc_loss +=  1. ** 2 *F.kl_div(stu_log_prob_m, tea_prob, reduction='batchmean')

            # Balanced Softmax calibration is shared by all experts.
            ce_loss += self.base_loss(logits[idx] +  self.log_prior, targets.repeat(2,))


        return ce_loss, sc_loss, ded_loss



def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))

    

def non_target_kd(logits_student, logits_teacher, target, temperature, cls_num_list, cb_trsfm=False):
    if cb_trsfm:
        coff = 0.9999
        coff_cls_wise = (1 - coff) / (1 - coff ** cls_num_list + 1e-5)
        coff_cls_wise = coff_cls_wise / coff_cls_wise.sum() * logits_student.shape[1]
    else:
        coff_cls_wise = torch.ones_like(cls_num_list)

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)


    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)

    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='none').sum(1)
        * (temperature**2) * coff_cls_wise[target]
    ).sum()
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )

    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='none').sum(1)
        * (temperature**2) * coff_cls_wise[target]
    ).sum()
    return tckd_loss + nckd_loss


def js_div(prob1, prob2):
    mean_prob = (prob1 + prob2) * 0.5
    loss = 0
    loss += F.kl_div(prob1.log(), mean_prob, reduction='batchmean')
    loss += F.kl_div(prob2.log(), mean_prob, reduction='batchmean')
    loss = loss * 0.5
    return loss


def dkd_loss(logits_student, logits_teacher, target, alpha=1.0, beta=1.0, temperature=1.0, cls_num_list=None, cb_trsfm=False):






    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')) * temperature ** 2
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )

    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean')) * temperature ** 2
    return tckd_loss*alpha + beta* nckd_loss


def intra_dkd_loss(logits_student, logits_teacher, target, temperature):
    gt_mask = _get_gt_mask(logits_student, target).T
    other_mask = _get_other_mask(logits_student, target).T
    pred_student = F.softmax(logits_student.T / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher.T / temperature, dim=1)

    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='none').sum(1)
    ) * temperature ** 2
    pred_teacher_part2 = F.softmax(
        logits_teacher.T / temperature - 1000.0 * gt_mask, dim=1
    )

    log_pred_student_part2 = F.log_softmax(
        logits_student.T / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='none').sum(1)
    ) * temperature ** 2

    tckd_loss = torch.nan_to_num(tckd_loss).mean()
    nckd_loss = torch.nan_to_num(nckd_loss).mean()

    return tckd_loss + nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt






    
