import torch
from torch.utils.tensorboard import SummaryWriter

class TensorBoard(object):
    def __init__(self):
        self.inst = None
    
    def init_path(self, path):
        self.inst = SummaryWriter(path)

    def close(self):
        self.inst.close()

# export singleton 
writer = TensorBoard()