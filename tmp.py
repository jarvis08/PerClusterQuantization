import torch


import torch
import torch.nn as nn
from time import time_ns

from HAWQ.utils.quantization_utils.quant_modules import *

class LinearModel(nn.Module):
  def __init__(self, channel, per_channel=False, per_batch=False, num_clusters=1):
    super().__init__()
    
    linear = nn.Linear(4096, channel, bias=True)
    self.layer = QuantLinear(bias_bit=32, per_channel=per_channel)
    self.layer.set_param(linear)
    
    if not per_batch:
      self.act = QuantAct()
    else:
      self.act = QuantAct_New(num_clusters=num_clusters)
    self.act.quant_model = "asymmetric"
  
  def forward(self, x, act_scaling_factor, cluster=None):
    x, weight_scaling_factor = self.layer(x, act_scaling_factor)
    x = nn.ReLU()(x)
    x, act_scaling_factor = self.act(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
    return x, weight_scaling_factor, act_scaling_factor
  
  
class ConvModel(nn.Module):
  def __init__(self, channel, per_channel=False, per_batch=False, num_clusters=1):
    super().__init__()
    
    conv = nn.Conv2d(256, channel, kernel_size=3, stride=2, padding=1, bias=True)
    self.layer = QuantConv2d(bias_bit=32, per_channel=per_channel)
    self.layer.set_param(conv)
    
    if not per_batch:
      self.act = QuantAct()
    else:
      self.act = QuantAct_New(num_clusters=num_clusters)
    self.act.quant_model = "asymmetric"
    
  def forward(self, x, act_scaling_factor, cluster=None):
    x, weight_scaling_factor = self.layer(x, act_scaling_factor)
    x = nn.ReLU()(x)
    x, act_scaling_factor = self.act(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
    return x, weight_scaling_factor, act_scaling_factor
  
  
def temp_fucntion(model, model_input, act_scaling_factor, cluster, trial):
  average = 0.0
  for i in range(100):
    start_time = time_ns()
    _ = model(model_input, act_scaling_factor, cluster)
    torch.cuda.synchronize()
    end_time = time_ns()
    
    average += (end_time - start_time) / 1000
    # print(f"time : {end_time - start_time}")

  return average / 100


def test(BATCH, CHANNEL, per_channel=False, per_batch=False, num_clusters=1, linear=True):
  
  if linear:
    data = torch.randn((BATCH,4096)).cuda()
    model = LinearModel(channel=CHANNEL, per_channel=per_channel, per_batch=per_batch, num_clusters=num_clusters).cuda()
  else :
    data = torch.randn((BATCH, 256, 16, 16)).cuda()
    model = ConvModel(channel=CHANNEL, per_channel=per_channel, per_batch=per_batch, num_clusters=num_clusters).cuda()
    
  if per_batch:
    cluster = torch.randint(0, num_clusters, (BATCH,)).cuda()
    quant_act = QuantAct_New(num_clusters=num_clusters).cuda()
  else:
    cluster = torch.randint(0, 1, (BATCH,)).cuda()
    quant_act = QuantAct().cuda()

  model_input, act_scaling_factor = quant_act(data, cluster=cluster)
  
  _ = temp_fucntion(model, model_input, act_scaling_factor, cluster, -1)
  torch.cuda.synchronize()
  
  average = 0
  
  for trial in range(10):
    average += temp_fucntion(model, model_input, act_scaling_factor, cluster, trial)
    
  return average / 10


def main():
  # data = torch.randn((32,256, 16, 16)).cuda()
  # cluster = torch.randint(0, 10, (32,)).cuda()
  # model = ConvModel(channel=128, per_channel=True, per_batch=True).cuda()
  
  # quant_act = QuantAct_New().cuda()
  # model_input, act_scaling_factor = quant_act(data, cluster=cluster)
  
  # tmp1, tmp2, tmp3 = model(model_input, act_scaling_factor, cluster=cluster)
  
  PER_CHANNEL = [False, True]
  PER_BATCH = [False, True]
  CLUSTERS = [1, 10]
  BATCH = [32, 64, 128, 256]
  
  CHANNEL = [512, 1024, 2048, 4096]
  for per_batch, num_clusters in zip(PER_BATCH, CLUSTERS):
    for per_channel in PER_CHANNEL:
      for batch in BATCH:
        for channel in CHANNEL:
          average = test(batch, channel, per_channel, per_batch, num_clusters, True)
          print(f"Batch : {batch}, Channel : {channel}, Per_channel : {per_channel}, Per_batch : {per_batch}, Linear, Total average time : {average}")
        print()
  
  CHANNEL = [128, 256, 512, 1024]
  for per_batch, num_clusters in zip(PER_BATCH, CLUSTERS):
    for per_channel in PER_CHANNEL:
      for batch in BATCH:
        for channel in CHANNEL:
          average = test(batch, channel, per_channel, per_batch, num_clusters, False)
          print(f"Batch : {batch}, Channel : {channel}, Per_channel : {per_channel}, Per_batch : {per_batch}, Conv2d, Total average time : {average}")
        print()
        
if __name__ == '__main__':
  main()