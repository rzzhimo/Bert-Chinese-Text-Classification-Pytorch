#这段代码时是为了测试本地机器的GPU是否可用，以我的机器Mac studio M1 MAX为例

import torch
print(torch.cuda.is_available())#会输出False
device = torch.device('mps')
print('Current device:', device)#会输出：Current device: mps
