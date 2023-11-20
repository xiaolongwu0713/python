import numpy as np
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./del")
r = 5
for i in range(100):
    writer.add_scalar('test1', i*np.sin(i/r), i)
    if i % 2 ==0:
        writer.add_scalar('test2', i * np.sin(i / r), i)
writer.close()