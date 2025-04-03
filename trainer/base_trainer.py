from models import Siren, PEMLP, Finer, Wire, Gauss
from util.logger import log
from util.tensorboard import writer
from util.misc import gen_cur_time
from util.recorder import recorder
from util.io import cvt_all_tensor
from components.transform import Transform

import numpy as np
import imageio.v2 as imageio
import torch
import torch.nn.functional as F
import os
from tqdm import trange
from collections import defaultdict

class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(self.args.device)
        self.data_name = os.path.splitext(os.path.basename(self.args.input_path))[0]
        self.transform = Transform(args)

        recorder.dic[self.data_name] = defaultdict(dict)
        self.recorder = recorder.dic[self.data_name]

    def _get_model(self, in_features, out_features):

        model_type = self.args.model_type

        if model_type == "siren":
            model = Siren(in_features=in_features, out_features=out_features, hidden_layers=self.args.hidden_layers, hidden_features=self.args.hidden_features,
                        first_omega_0=self.args.first_omega, hidden_omega_0=self.args.hidden_omega)
        elif model_type == 'pemlp':
            model = PEMLP(in_features=in_features, out_features=out_features, hidden_layers=self.args.hidden_layers, hidden_features=self.args.hidden_features,
                        N_freqs=self.args.N_freqs)
        elif model_type == 'finer':
            model = Finer(in_features=in_features, out_features=out_features, hidden_layers=self.args.hidden_layers, hidden_features=self.args.hidden_features,
                        first_omega_0=self.args.first_omega, hidden_omega_0=self.args.hidden_omega, 
                        first_bias_scale=self.args.first_bias_scale, scale_req_grad=self.args.scale_req_grad) # specific for FINER
        elif model_type == 'wire':
            model = Wire(in_features=in_features, out_features=out_features, hidden_layers=self.args.hidden_layers, hidden_features=self.args.hidden_features,
                        first_omega_0=self.args.first_omega, hidden_omega_0=self.args.hidden_omega, scale=self.args.scale)
        elif model_type == 'gauss':
            model = Gauss(in_features=in_features, out_features=out_features, hidden_layers=self.args.hidden_layers, hidden_features=self.args.hidden_features,
                        scale=self.args.scale)
        else:
            model = None
        return model
        
    def _get_data(self):
        return NotImplementedError

    def train(self):
        raise NotImplementedError
        
    def _save_ckpt(self, epoch, model, optimizer, scheduler):
        state_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            # 'best_score': 0,
            # 'final_score': 0,
            'args': self.args
        }
        self._save_ckpt_by_dict(state_dict, epoch)
    
    def _save_ckpt_by_dict(self, dic:dict, epoch):
        cur_time = gen_cur_time()
        saved_name =  f"ckpt_{self.data_name}_{epoch}_{cur_time}.pth"
        torch.save(dic, self._get_sub_path("ckpt", saved_name))
        log.inst.success(f"Save ckpt in epoch {epoch} named {saved_name}")

    def _get_cur_path(self, filename):
        return os.path.join(self.args.save_folder, filename)
    
    def _get_sub_path(self, subfolder, filename):
        saved_path = self._mk_sub_path(subfolder)
        return os.path.join(saved_path, filename)
    
    def _mk_sub_path(self, subfolder):
        sub_path = os.path.join(self.args.save_folder,subfolder)
        os.makedirs(sub_path, exist_ok=True)
        return sub_path
        

    # wrapper
    def encode_zero_mean(self, data):
        return self._encode_zero_mean(data)
    
    def decode_zero_mean(self,data):
        return self._decode_zero_mean(data)

    @staticmethod
    def _encode_zero_mean(data):
        return (data - 0.5) / 0.5

    @staticmethod
    def _decode_zero_mean(data):
        return data/2 + 0.5
    
    @staticmethod
    def compute_mae(pred, gt):
        # return torch.abs((pred - gt)).mean()
        return F.l1_loss(pred, gt)

    @staticmethod
    def compute_mse(pred, gt):
        #  return torch.mean((pred - gt) ** 2)
        return F.mse_loss(pred, gt)
       