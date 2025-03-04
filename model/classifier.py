'''
Author: sanctuary 836790896@qq.com
Date: 2024-10-05 10:30:50
LastEditors: sanctuary 836790896@qq.com
LastEditTime: 2024-10-05 10:31:57
FilePath: /ecg_self_supervised_training/model/classifier.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch
from torch import nn

class Classifier(nn.Module):

    def __init__(self, pre_train_model, classifier_head):
        super().__init__()
        self.pre_train_model = pre_train_model
        self.classifier_head = classifier_head
    
    @property
    def name(self):
        return f'{self.pre_train_model.name}+{self.classifier_head.name}'
    
    def forward(self, x):

        embedding = self.pre_train_model.forward_feature(x)
        out = self.classifier_head(embedding)
        return out
    
# class moving_avg(nn.Module):
#     """
#     Moving average block to highlight the trend of time series
#     """
#     def __init__(self, kernel_size, stride):
#         super(moving_avg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

#     def forward(self, x):
#         # padding on the both ends of time series
#         front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         x = torch.cat([front, x, end], dim=1)
#         x = self.avg(x.permute(0, 2, 1))
#         x = x.permute(0, 2, 1)
#         return x


# class series_decomp(nn.Module):
#     """
#     Series decomposition block
#     """
#     def __init__(self, kernel_size):
#         super(series_decomp, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)

#     def forward(self, x):
#         moving_mean = self.moving_avg(x)
#         res = x - moving_mean
#         return res, moving_mean
    
# class DecomClassifier(nn.Module):

#     def __init__(self, pre_train_model, pretrain_out_dim):   
#         super().__init__()
#         self.pre_train_model = pre_train_model
#         self.pretrain_out_dim = pretrain_out_dim

#         # Decompsition Kernel Size
#         kernel_size = 25
#         self.decompsition = series_decomp(kernel_size)
#         self.channels = 1

#         self.Linear_Seasonal = nn.Linear(1200, 1)
#         self.Linear_Trend = nn.Linear(1200, 1)

#         self.Linear_Seasonal.weight = nn.Parameter((1/1200)*torch.ones([1,1200]))
#         self.Linear_Trend.weight = nn.Parameter((1/1200)*torch.ones([1,1200]))

#         self.fc1 = nn.Linear(2 + 1200 + self.pretrain_out_dim, 2 + 1200 + self.pretrain_out_dim)
#         self.relu = torch.nn.ReLU()
#         self.fc2 = nn.Linear(2 + 1200 + self.pretrain_out_dim, 4)
    
#     @property
#     def name(self):
#         return f'{self.pre_train_model.name}+decom_classifier'
    
#     def forward(self, x):
#         embedding = self.pre_train_model.forward_feature(x)

#         x = x.permute(0,2,1)
#         seasonal_init, trend_init = self.decompsition(x)
#         seasonal_init, trend_init = seasonal_init.squeeze(-1), trend_init.squeeze(-1)
#         x = x.squeeze(-1)

#         seasonal_output = self.Linear_Seasonal(seasonal_init)
#         trend_output = self.Linear_Trend(trend_init)

#         out = torch.cat([x, seasonal_output, trend_output, embedding], dim=1)
#         out = self.fc1(out)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out



