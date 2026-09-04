# Portions derive from Balanced Contrastive Learning; see
# LICENSES/BALANCED_CONTRASTIVE_LEARNING.txt in the repository root.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.tensor(cls_num_list).cuda()
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_neg = torch.zeros_like(x, dtype=torch.uint8)
        mask = index_neg.eq(index).byte()
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, reduction='mean')


class VSLoss(nn.Module):

    def __init__(self, cls_num_list, gamma=0.2, tau=1.2, weight=None):
        super(VSLoss, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        temp = (1.0 / np.array(cls_num_list)) ** gamma
        temp = temp / np.min(temp)

        iota_list = tau * np.log(cls_probs)
        Delta_list = temp

        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.Delta_list = torch.cuda.FloatTensor(Delta_list)
        self.weight = weight

    def forward(self, x, target):
        output = x / self.Delta_list + self.iota_list

        return F.cross_entropy(output, target, weight=self.weight)


class HellingDistance(nn.Module):
    def __init__(self, cls_num_list, tau=1, temp=0.2):
        super(HellingDistance, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.tau = tau
        self.temp = temp

    def forward(self, x, target, la=False):
        device = x.device
        if la:
            x = x + self.tau * self.m_list
        p = F.one_hot(target.view(-1,), num_classes=self.m_list.shape[1]).to(torch.float32)
        q = F.softmax(x, dim=1)
        print(q)
        loss = (p - q.sqrt()).pow(2).sum(1) / 2
        loss = loss.mean()
        return loss


class NormalizeClf(nn.Module):
    def __init__(self, cls_num_list, tau=1, temp=0.2):
        super(NormalizeClf, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.tau = tau
        self.temp = temp

    def forward(self, x, target, la=False):
        device = x.device
        if la:
            x = x + self.tau * self.m_list
        q = F.one_hot(target.view(-1, ), num_classes=self.m_list.shape[1]).to(torch.float32)
        p = x.softmax(1).log()
        loss = F.kl_div(p, q, reduction='batchmean')
        return loss


class DiverseExpertLoss(nn.Module):
    def __init__(self, cls_num_list=None,  prior_list=None):
        super().__init__()
        self.base_loss = F.cross_entropy 
        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number
        self.prior_list = prior_list 
        

    def forward(self, output_logits, logits_k, target):
        loss = 0
        batch_size = target.shape[0]

        expert1_logits = output_logits[0]
        expert2_logits = output_logits[1]
        expert3_logits = output_logits[2]

        expert1_logits = expert1_logits + torch.log(self.prior + 1e-9) * self.prior_list[0] 
        loss += self.base_loss(expert1_logits[:batch_size], target)
        
        expert2_logits = expert2_logits + torch.log(self.prior + 1e-9) * self.prior_list[1] 
        loss += self.base_loss(expert2_logits[:batch_size], target)
        
        expert3_logits = expert3_logits + torch.log(self.prior + 1e-9) * self.prior_list[2] 
        loss += self.base_loss(expert3_logits[:batch_size], target)


        




        

        return loss
    

class MultiExpertLoss(nn.Module):
    def __init__(self, cls_num_list=None,  max_m=0.5, s=30, tau=2):
        super().__init__()
        self.base_loss = F.cross_entropy 
     
        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = tau 

    def inverse_prior(self, prior): 
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0]-1-idx1 # reverse the order
        inverse_prior = value.index_select(0,idx2)
        
        return inverse_prior

    def forward(self, output_logits, target, extra_info=None):

        loss = 0
        batch_size = target.shape[0]

        expert1_logits = output_logits[0][:batch_size]
        expert2_logits = output_logits[1][:batch_size]
        expert3_logits = output_logits[2][:batch_size]

        loss += self.base_loss(expert1_logits, target)
        
        expert2_logits = expert2_logits + torch.log(self.prior + 1e-9) 
        loss += self.base_loss(expert2_logits, target)
        
        inverse_prior = self.inverse_prior(self.prior)
        expert3_logits = expert3_logits + torch.log(self.prior + 1e-9) - self.tau * torch.log(inverse_prior+ 1e-9) 
        loss += self.base_loss(expert3_logits, target)
   
        return loss
    

class MDCSLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, tau=2):
        super().__init__()
        self.base_loss = F.cross_entropy

        prior = np.array(cls_num_list) #/ np.sum(cls_num_list)

        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = 2

        self.additional_diversity_factor = -0.2
        out_dim = 100
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center1", torch.zeros(1, out_dim))
        self.center_momentum = 0.9
        self.warmup = 20  
        self.reweight_epoch = 200
        if self.reweight_epoch != -1:
            idx = 1  # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float,
                                                        requires_grad=False)  # 这个是logits时算CE loss的weight
        self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float,
                                                              requires_grad=False).cuda()  # 这个是logits时算diversity loss的weight



    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def forward(self, output_logits, target, epoch):
    
        loss = 0
        temperature_mean = 1
        temperature = 1  
        num = target.shape[0]

        expert1_logits = output_logits[0] + torch.log(torch.pow(self.prior, -0.5) + 1e-9)      #head

        expert2_logits = output_logits[1] + torch.log(torch.pow(self.prior, 1) + 1e-9)         #medium

        expert3_logits = output_logits[2] + torch.log(torch.pow(self.prior, 2.5) + 1e-9)       #few



        teacher_expert1_logits = expert1_logits[num:, :]  # view1
        student_expert1_logits = expert1_logits[:num, :]  # view2

        teacher_expert2_logits = expert2_logits[num:, :]  # view1
        student_expert2_logits = expert2_logits[:num, :]  # view2

        teacher_expert3_logits = expert3_logits[num:, :]  # view1
        student_expert3_logits = expert3_logits[:num, :]  # view2



        teacher_expert1_softmax = F.softmax((teacher_expert1_logits) / temperature, dim=1).detach()
        student_expert1_softmax = F.log_softmax(student_expert1_logits / temperature, dim=1)

        teacher_expert2_softmax = F.softmax((teacher_expert2_logits) / temperature, dim=1).detach()
        student_expert2_softmax = F.log_softmax(student_expert2_logits / temperature, dim=1)

        teacher_expert3_softmax = F.softmax((teacher_expert3_logits) / temperature, dim=1).detach()
        student_expert3_softmax = F.log_softmax(student_expert3_logits / temperature, dim=1)


         

        teacher1_max, teacher1_index = torch.max(F.softmax((teacher_expert1_logits), dim=1).detach(), dim=1)
        student1_max, student1_index = torch.max(F.softmax((student_expert1_logits), dim=1).detach(), dim=1)

        teacher2_max, teacher2_index = torch.max(F.softmax((teacher_expert2_logits), dim=1).detach(), dim=1)
        student2_max, student2_index = torch.max(F.softmax((student_expert2_logits), dim=1).detach(), dim=1)

        teacher3_max, teacher3_index = torch.max(F.softmax((teacher_expert3_logits), dim=1).detach(), dim=1)
        student3_max, student3_index = torch.max(F.softmax((student_expert3_logits), dim=1).detach(), dim=1)


        partial_target = target[:num]
        kl_loss = 0
     
        if torch.sum((teacher1_index == partial_target)) > 0:
            kl_loss = kl_loss + F.kl_div(student_expert1_softmax[(teacher1_index == partial_target)],
                                         teacher_expert1_softmax[(teacher1_index == partial_target)],
                                         reduction='batchmean') * (temperature ** 2)

        if torch.sum((teacher2_index == partial_target)) > 0:
            kl_loss = kl_loss + F.kl_div(student_expert2_softmax[(teacher2_index == partial_target)],
                                         teacher_expert2_softmax[(teacher2_index == partial_target)],
                                         reduction='batchmean') * (temperature ** 2)

        if torch.sum((teacher3_index == partial_target)) > 0:
            kl_loss = kl_loss + F.kl_div(student_expert3_softmax[(teacher3_index == partial_target)],
                                         teacher_expert3_softmax[(teacher3_index == partial_target)],
                                         reduction='batchmean') * (temperature ** 2)




        loss += self.base_loss(expert1_logits, target.repeat(2,))

        loss += self.base_loss(expert2_logits, target.repeat(2,))

        loss += self.base_loss(expert3_logits, target.repeat(2,))


        return loss + kl_loss*min(epoch / 20, 1)*0.6
    

class MultiExpertCL(nn.Module):
    def __init__(self, cls_num_list=None, temp=0.1, m=0.99, output_dim=1024):
        super().__init__()
        self.base_loss = F.cross_entropy
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_prior = cls_num_list / cls_num_list.sum()
        self.cls_log_prior = torch.log(cls_prior + 1e-9).view(1, -1)
        self.temp = temp
        self.m = m
        self.register_buffer("center_e1", torch.zeros(len(cls_prior), output_dim))
        self.register_buffer("center_e2", torch.zeros(len(cls_prior), output_dim))
        self.register_buffer("center_e3", torch.zeros(len(cls_prior), output_dim))

    def forward(self, feature, targets):
        batch_size = targets.shape[0]
        feat_e1 = feature[0]
        feat_e2 = feature[1]
        feat_e3 = feature[2]


        for f1, f2, f3, label in zip(feat_e1, feat_e2, feat_e3, targets.repeat(2,)):
            self.center_e1[label] = self.center_e1[label] * self.m * f1 * (1 - self.m)
            self.center_e2[label] = self.center_e2[label] * self.m * f2 * (1 - self.m)
            self.center_e3[label] = self.center_e3[label] * self.m * f3 * (1 - self.m)
        
        self.center_e1 = F.normalize(self.center_e1, dim=1)
        self.center_e2 = F.normalize(self.center_e2, dim=1)
        self.center_e3 = F.normalize(self.center_e3, dim=1)
        
        proto_e1 = self.center_e1.clone().detach()
        proto_e2 = self.center_e2.clone().detach()
        proto_e3 = self.center_e3.clone().detach()

        loss = 0

        logits_e1 = feat_e1.mm(proto_e1.T) / self.temp
        loss += self.base_loss(logits_e1, targets.repeat(2,))

        logits_e2 = feat_e2.mm(proto_e2.T) / self.temp
        loss += self.base_loss(logits_e2, targets.repeat(2,))

        logits_e3 = feat_e3.mm(proto_e3.T) / self.temp
        loss += self.base_loss(logits_e3, targets.repeat(2,))
        
        return loss

    @torch.no_grad()
    def update_center(self, feat_e1, feat_e2, feat_e3, targets):
        """
        Update center used for teacher output.
        """

        for f1, f2, f3, label in zip(feat_e1, feat_e2, feat_e3, targets):
            self.center_e1[label] = self.center_e1[label] * self.m * f1 * (1 - self.m)
            self.center_e2[label] = self.center_e2[label] * self.m * f2 * (1 - self.m)
            self.center_e3[label] = self.center_e3[label] * self.m * f3 * (1 - self.m)
        
        self.center_e1 = F.normalize(self.center_e1, dim=1)
        self.center_e2 = F.normalize(self.center_e2, dim=1)
        self.center_e3 = F.normalize(self.center_e3, dim=1)




class ExpertKDLoss(nn.Module):
    def __init__(self, cls_num_list=None,  prior_list=None, kd_type='mutual', temp=1.0):
        super().__init__()
        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number
        self.prior_list = prior_list 
        self.kd_type = kd_type
        self.temp = temp

        

    def forward(self, logits_stu, logits_tea, target):
        ce_loss = 0
        kl_loss = 0
        batch_size = target.shape[0]

        expert1_logits_stu = logits_stu[0] + self.prior_list[0] * torch.log(self.prior + 1e-9)
        expert2_logits_stu = logits_stu[1] + self.prior_list[1] * torch.log(self.prior + 1e-9)
        expert3_logits_stu = logits_stu[2] + self.prior_list[2] * torch.log(self.prior + 1e-9)

        expert1_logits_tea = logits_tea[0] + self.prior_list[0] * torch.log(self.prior + 1e-9)
        expert2_logits_tea = logits_tea[1] + self.prior_list[1] * torch.log(self.prior + 1e-9)
        expert3_logits_tea = logits_tea[2] + self.prior_list[2] * torch.log(self.prior + 1e-9)

        ce_loss += F.cross_entropy(expert1_logits_stu, target.repeat(2,))
        ce_loss += F.cross_entropy(expert2_logits_stu, target.repeat(2,))
        ce_loss += F.cross_entropy(expert3_logits_stu, target.repeat(2,))


        expert_logits_stu_list = [expert1_logits_stu, expert2_logits_stu, expert3_logits_stu]
        expert_logits_tea_list = [expert1_logits_tea, expert2_logits_tea, expert3_logits_tea]
        
        if self.kd_type == 'mutual':
            for i in range(3):
                dist_tea = F.softmax(expert_logits_stu_list[i], dim=1).detach()
                dist_tea_v1, dist_tea_v2 = dist_tea.chunk(2, dim=0)
                dist_tea = torch.cat([dist_tea_v2, dist_tea_v1], dim=0)
                _, teacher_index = torch.max(dist_tea, dim=1)
                for j in range(3):
                    if i == j:
                        continue
                    dist_stu = F.log_softmax(expert_logits_stu_list[j], dim=1)
                    kl_loss += F.kl_div(dist_stu, dist_tea.detach(), reduction='batchmean')
            kl_loss = kl_loss / 6

        elif self.kd_type == 'mutual-dis':
            for i in range(3):
                dist_tea = F.softmax(expert_logits_tea_list[i], dim=1).detach()
                _, teacher_index = torch.max(dist_tea, dim=1)
                for j in range(3):
                    if i == j:
                        continue
                    dist_stu = F.log_softmax(expert_logits_stu_list[j], dim=1)
                    kl_loss += F.kl_div(dist_stu, dist_tea, reduction='batchmean')
            kl_loss = kl_loss / 6

        elif self.kd_type == 'self-dis':

            for stu, tea in zip(expert_logits_stu_list, expert_logits_tea_list):
                dist_tea = F.softmax(tea, dim=1).detach()
                _, teacher_index = torch.max(dist_tea, dim=1)
                dist_stu = F.log_softmax(stu, dim=1)
                kl_loss += F.kl_div(dist_stu, dist_tea.detach(), reduction='batchmean')
            kl_loss = kl_loss / 3

        elif self.kd_type == 'self-aggre':
            dist_tea = (expert1_logits_tea + expert2_logits_tea + expert3_logits_tea) / 3
            dist_tea = F.softmax(dist_tea, dim=1).detach()


            _, teacher_index = torch.max(dist_tea, dim=1)
            for stu in expert_logits_stu_list:
                dist_stu = F.log_softmax(stu, dim=1)
                kl_loss += F.kl_div(dist_stu, dist_tea.detach(), reduction='batchmean')
            kl_loss = kl_loss / 3

        elif self.kd_type == 'self-aggre-mutual':
            dist_tea = (expert1_logits_tea + expert2_logits_tea + expert3_logits_tea) / 3
            dist_tea = F.softmax(dist_tea, dim=1).detach()
            _, teacher_index = torch.max(dist_tea, dim=1)
            for stu in expert_logits_stu_list:
                dist_stu = F.log_softmax(stu, dim=1)
                kl_loss += F.kl_div(dist_stu[teacher_index == target], dist_tea[teacher_index == target].detach(), reduction='batchmean') / 3
            
            for i in range(3):
                dist_tea = F.softmax(expert_logits_stu_list[i], dim=1).detach()
                _, teacher_index = torch.max(dist_tea, dim=1)
                for j in range(3):
                    if i == j:
                        continue
                    dist_stu = F.log_softmax(expert_logits_stu_list[j], dim=1)
                    kl_loss += F.kl_div(dist_stu[teacher_index == target], dist_tea[teacher_index == target].detach(), reduction='batchmean') / 6
           
        return ce_loss, kl_loss
    

class MixedCEKDLoss(nn.Module):
    def __init__(self, cls_num_list=None,  temp=1.0):
        super().__init__()
        self.cls_num_list = torch.tensor(cls_num_list).cuda()
        self.log_prior = torch.log(self.cls_num_list / self.cls_num_list.sum())
        self.C_number = len(cls_num_list)  # class number
        self.base_loss = F.cross_entropy
        self.temp = temp

    def forward(self, logits, logits_k, logits_s, logits_m, targets):

        ce_loss = 0
        kl_loss = 0
        kl_ens_loss = 0

        batch_size = targets.shape[0]

        expert1_logits_ce = logits[0] +  self.log_prior
        expert2_logits_ce = logits[1] +  self.log_prior
        expert3_logits_ce = logits[2] +  self.log_prior
            
        expert1_logits_m = logits_m[0] +  self.log_prior
        expert2_logits_m = logits_m[1] +  self.log_prior
        expert3_logits_m = logits_m[2] +  self.log_prior

        expert1_logits_t = logits_k[0] + self.log_prior
        expert2_logits_t = logits_k[1] + self.log_prior
        expert3_logits_t = logits_k[2] + self.log_prior
        
        expert1_logits_s = logits_s[0] + self.log_prior
        expert2_logits_s = logits_s[1] + self.log_prior
        expert3_logits_s = logits_s[2] + self.log_prior

        ce_loss += self.base_loss(expert1_logits_ce, targets.repeat(2,))
        ce_loss += self.base_loss(expert2_logits_ce, targets.repeat(2,))
        ce_loss += self.base_loss(expert3_logits_ce, targets.repeat(2,))
        
        stu_list_m = [expert1_logits_m, expert2_logits_m, expert3_logits_m]
        stu_list_s = [expert1_logits_s, expert2_logits_s, expert3_logits_s]
        tea_list = [expert1_logits_t, expert2_logits_t, expert3_logits_t]
        

        tea_ens_all = (tea_list[0] + tea_list[1] + tea_list[2]).detach()
        tea_ens_v1, tea_ens_v2 = tea_ens_all.chunk(2, dim=0)
        tea_ens = (tea_ens_v1 + tea_ens_v2) / 6
        
        tea_label = torch.max(tea_ens.detach(), 1)[1]

        for i in range(3):
            stu_label = torch.max(stu_list_s[i].detach(), 1)[1]
            
            select_index = (tea_label != targets) & (stu_label == targets)
            select_index = ~select_index
            
            if torch.sum(select_index) > 0:
                kl_ens_loss += self.temp ** 2 *F.kl_div(F.log_softmax(stu_list_s[i][select_index] / self.temp, dim=1),
                             F.softmax(tea_ens[select_index] / self.temp, dim=1), reduction='batchmean')
            

            tea_v1, tea_v2 = tea_list[i].detach().chunk(2, dim=0)
            tea_mean = (tea_v1 + tea_v2) / 2
            tea_prob = F.softmax(tea_mean / 1., dim=1)
            stu_log_prob = F.log_softmax(stu_list_m[i] / 1., dim=1)

            kl_loss +=  1. ** 2 *F.kl_div(stu_log_prob, tea_prob, reduction='batchmean')   
       
        return ce_loss, kl_loss, kl_ens_loss
    

class MixedSelfDis(nn.Module):
    def __init__(self, cls_num_list=None, cls_nums=100, temp=1.0):
        super().__init__()
        if cls_num_list is not None:
            self.cls_num_list = torch.tensor(cls_num_list).cuda()
            self.cls_prior = self.cls_num_list / self.cls_num_list.sum()
            self.log_prior = self.cls_prior.log()
        else:
            self.log_prior = torch.tensor([0.]).cuda()
        self.C_number = cls_nums
        self.temp = temp
        self.base_loss = F.cross_entropy
    

    def forward(self, logits, logits_k, targets, epoch):
        ce_loss = 0
        sd_loss = 0
        ml_loss = 0

        batch_size = targets.shape[0]
        num_experts = logits.shape[0]
        
        for i in range(num_experts):
            logits_stu = logits[i] + self.log_prior
            ce_loss += self.base_loss(logits_stu, targets.repeat(2,))
            prob_stu_sd = F.softmax(logits_stu / self.temp, dim=1)
            _, stu_index = torch.max(logits_stu, dim=1)

            with torch.no_grad():
                logits_tea = logits_k[i] + self.log_prior
                prob_tea_sd = F.softmax(logits_tea / self.temp, dim=1)
            
            
            prob_stu_ml = F.softmax(logits_stu / 1.5, dim=1)
            
            prob_stu_ml_ent = F.softmax(logits_stu / 1.5, dim=1).detach()
            ent_stu = -(prob_stu_ml_ent * prob_stu_ml_ent.log()).sum(1)

            for j in range(num_experts):
                
                with torch.no_grad():
                    logits_ml = logits_k[j] + self.log_prior
                    logits_ml_v1, logits_ml_v2 = logits_ml.chunk(2, dim=0)
                    logits_ml = torch.cat([logits_ml_v2, logits_ml_v1], dim=0)
                    prob_ml = F.softmax(logits_ml / 1.5, dim=1)
                
                ml_loss += 1.5 ** 2 * F.kl_div(F.log_softmax(logits_stu / 1.5, dim=1),
                        prob_ml, reduction='batchmean'
                        )


                



                

                

                    
                
    
        return ce_loss, sd_loss, ml_loss


class KDCEoss(nn.Module):
    def __init__(self, cls_num_list=None, temp=1.0):
        super().__init__()
        if cls_num_list is not None:
            prior = np.array(cls_num_list) / np.sum(cls_num_list)
            self.cls_num_list = torch.cuda.FloatTensor(cls_num_list)
            self.log_prior = torch.cuda.FloatTensor(prior).log()
            self.C_number = len(cls_num_list)  # class number
        self.temp = temp
        self.base_loss = F.cross_entropy
    
    def forward(self, logits, logits_k, targets, epoch):
        batch_size = targets.shape[0]
        num_experts = logits.shape[0]
        
        ce_loss = 0
        kl_consis_loss = 0
        kl_div_loss = 0
        kl_loss = 0
        
        kl_ob_loss = 0




                        


        
        tau_list = [-1, 1, 3]


        for i in range(num_experts):
            logits_stu = logits[i]+ self.log_prior 

            log_prob_stu = F.log_softmax(logits_stu / 1.5, dim=1)


            logits_tea = (logits_k[i] + self.log_prior).repeat(2, 1)
            
            tea_softmax = F.softmax(logits_tea / self.temp, dim=1).detach()

            stu_log_softmax = F.log_softmax(logits_stu / self.temp, dim=1)
            _, tea_index = torch.max(logits_tea, dim=1)
            _, stu_index = torch.max(logits_stu, dim=1)
            
            ent_softmax = F.softmax(logits_tea / 0.5, dim=1).detach()
            ent_tea = -(ent_softmax * ent_softmax.log()).sum(1)
            ent_stu_softmax = F.softmax(logits_stu / 0.5, dim=1).detach()
            ent_stu = -(ent_stu_softmax * ent_stu_softmax.log()).sum(1)
            
            
            ent_max = np.log(self.C_number)
            ce_loss += self.base_loss(logits_stu, targets.repeat(2,)) 
            



            
            stu_pred_prob = F.softmax(logits_stu / self.temp, dim=1).detach()

            tea_prob_select = tea_softmax[torch.arange(batch_size*2).cuda(), targets.repeat(2,)]
            stu_prob_select = stu_pred_prob[torch.arange(batch_size*2).cuda(), targets.repeat(2,)] 

            
            ent_coeff = 1 + 0.5* ( ent_stu / ent_max)

            select_index = (tea_index != targets.repeat(2,)) & (stu_index == targets.repeat(2,))
            select_index = ~select_index

            num_samples = torch.sum(select_index)

                
            



            

            



            
            
            
            ml_nums = 0
            common_index = 0
            

            with torch.no_grad():
                select_ml_all = torch.empty(batch_size * 2, 0).cuda()

                for ind_k in range(num_experts):
                    if i == ind_k:
                        continue

                    ml_logits = logits[ind_k] + self.log_prior
                    ml_logits_v1, ml_logits_v2 = ml_logits.chunk(2 ,dim=0)
                    ml_logits_cross = torch.cat([ml_logits_v2, ml_logits_v1], dim=0)
                    
                    _, ml_index = torch.max(ml_logits_cross, dim=1)
                    select_ml = ~((ml_index != targets.repeat(2,)) & (stu_index == targets.repeat(2,)))
                    select_ml = select_ml.float().view(-1, 1)
                    select_ml_all = torch.cat((select_ml_all, select_ml), dim=1)
                

                select_ml_all_sum = select_ml_all.sum(1, keepdim=True)
                amp_ml = torch.where(select_ml_all_sum != 0, (num_experts - 1) / select_ml_all_sum, torch.zeros_like(select_ml_all_sum))
                select_ml_all_final = select_ml_all * amp_ml
            
            select_ml_nums = select_ml_all_final.sum()
            amp_index = 0
            for j in range(num_experts):
                if i == j:
                    continue
                
                mutual_coeff = select_ml_all_final[:, amp_index]
                amp_index += 1

                prob_tea_logits = logits[j] + self.log_prior 
                prob_tea_logits_v1, prob_tea_logits_v2 = prob_tea_logits.chunk(2,dim=0)
                prob_tea_logits_cross = torch.cat([prob_tea_logits_v2, prob_tea_logits_v1], dim=0)
                prob_tea_consis = F.softmax(prob_tea_logits_cross / 1.5,dim=1).detach()
                prob_tea_consis2 = F.softmax(prob_tea_logits_cross / 0.5,dim=1).detach()

                _, tea_index2 = torch.max(prob_tea_consis, dim=1)
                select_index_mutual = (tea_index2 != targets.repeat(2,)) & (stu_index == targets.repeat(2,))
                select_index_mutual2 = (tea_index2 == targets.repeat(2,)) & (stu_index != targets.repeat(2,))

                select_index_mutual = ~select_index_mutual

                mutual_prob = prob_tea_consis[torch.arange(batch_size*2).cuda(), targets.repeat(2,)]

                ent_mutual = -(prob_tea_consis2 * prob_tea_consis2.log()).sum(1)
                ent_coeff2 = 0.5 * ( ent_tea / ent_max) 
                
                
                ent_coeff2 = 1 + ent_coeff2  

                common_index += (select_index_mutual).float()
                ml_coeff = select_index_mutual.float()
                kl_div_loss += 1.5*1.5*(F.kl_div(log_prob_stu, prob_tea_consis, reduction='none').sum(1) * mutual_coeff  ).sum() / mutual_coeff.sum() / (num_experts - 1)
                
            
            






        
            


            






            

            
            
        
        

        return ce_loss, kl_consis_loss, kl_div_loss
    


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






    
