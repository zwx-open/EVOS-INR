import lpips
import torch

class Calc_LPIPS(object):
    def __init__(self, net_name="alex", device="cpu"):
        assert net_name in ['alex', 'vgg']
        self.lpips_net =  lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

    def _norm(self, data):
        _min = torch.min(data)
        _max = torch.max(data)
        data = (data - _min) / (_max - _min) # [0,1]
        data = (data - 0.5) * 2  # [-0.5, 0.5]
        return data
    
    def compute_lpips(self, gt, pred):
        normed_gt = self._norm(gt).squeeze(0)
        normed_pred = self._norm(pred).squeeze(0)
        return self.lpips_net(normed_gt, normed_pred).detach().item()
    
    


