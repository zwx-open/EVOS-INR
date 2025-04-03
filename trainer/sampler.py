from components.nmt import NMT
from components.nmt import mt_scheduler_factory
from components.lmc import LMC
from components.evos import EVOS

import numpy as np
import torch

from util.misc import fix_seed


class Sampler(object):
    def __init__(self, args):
        self.args = args
        self._st = self.args.strategy
        self.use_ratio_scheduler = mt_scheduler_factory(self.args.sample_num_schedular)

    def _init_sampler(self):
        _st = self.args.strategy
        if _st == "nmt":
            self._nmt_init()
        elif _st == "evos":
            self._evos_init()
        elif _st == "soft":
            self._soft_init()

    def _nmt_init(self):
        if self.args.signal_type == "image":
            _dim = (self.H, self.W, self.C)
        else:
            _dim = (1,) # placeholder

        self.nmt = NMT(
            self.model,
            self.args.num_epochs,
            _dim,
            self.args.sample_num_schedular,  # "step" "incremental" "constant"
            self.args.nmt_profile_strategy,  # "dense" "incremental"
            self.args.use_ratio,
            True,
            #   save_samples_path=self._get_sub_path("nmt", "samples"),
            #   save_losses_path=self._get_sub_path("nmt", "losses"),
            save_samples_path=None,
            save_losses_path=None,
            save_name=None,
            save_interval=self.args.log_epoch,
        )

    def _soft_init(self):
        minpct = 0.1
        lossminpc = 0.1
        bs = np.ceil(self.args.use_ratio * self.H * self.W).astype(np.int32)
        images = self.input_img.unsqueeze(0).permute(0, 2, 3, 1)  # n,h,w,c
        const_img_id = torch.arange(
            0, images.shape[0], device=self.device
        ).repeat_interleave(bs)

        self.lmc = LMC(
            images=images,
            num_rays=bs,
            const_img_id=const_img_id,
            device=self.device,
            minpct=minpct,
            lossminpc=lossminpc,
            a=self.args.soft_mining_a,
            b=self.args.soft_mining_b,
        )
    
    def _evos_init(self, input_data=None):
        self.evos = EVOS(
            input_data=input_data, # only for image
            device=self.args.device,
            signal_type = self.args.signal_type,
            use_ratio=self.args.use_ratio,
            num_epochs=self.args.num_epochs,
            fitness_interval_method=self.args.profile_interval_method,
            init_interval=self.args.init_interval,
            end_interval=self.args.end_interval,
            high_freq_lamda=self.args.lap_coff,
            mutation_method=self.args.mutation_method,
            init_mutation_ratio=self.args.init_mutation_ratio,
            end_mutation_ratio=self.args.end_mutation_ratio,
            crossover_method=self.args.crossover_method,
            use_cfs_epoch=self.args.use_laplace_epoch,
        )
    
    def _get_cur_use_ratio(self, epoch):
        return self.use_ratio_scheduler(
            epoch, self.args.num_epochs, self.args.use_ratio
        )

    def _reset_randomness(self):
        generator = torch.Generator()
        seed = generator.seed()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _reset_fixed_seed(self):
        fix_seed(self.args.seed)


    def get_sampled_coords_gt(self, coords, gt, epoch, **kw):
        
        self.sample_num = coords.shape[0]  ### full_coords

        if self._st == "full":
            return coords, gt
        else:
            self._reset_randomness()
            cur_use_ratio = self._get_cur_use_ratio(epoch)
            if self._st == "random":
                _ratio_len = int(self.sample_num * cur_use_ratio)
                indices = torch.randperm(self.sample_num, device=self.device)[
                    :_ratio_len
                ]
                sampled_coords = coords[indices]
                sampled_gt = gt[indices]
            
            elif self._st == "nmt":
                sampled_coords, sampled_gt, _, indices = self.nmt.sample(
                    epoch - 1, coords, gt
                )
            elif self._st == "evos":
                if self.args.signal_type == "shape":
                    sampled_coords, sampled_gt = self.evos.select_sample_via_index(coords, gt, epoch, cur_use_ratio, **kw)
                else:
                    sampled_coords, sampled_gt = self.evos.select_sample(coords, gt, epoch, cur_use_ratio)
            self._reset_fixed_seed()
            return sampled_coords, sampled_gt
    
    def sampler_operate_after_pred(self, pred, gt, epoch, recons_func):
        if self._st == "evos":
            if self.evos._is_fitness_eval_iter(epoch):
                self.evos.crossover(pred, gt, epoch, recons_func)
            
        elif self._st == "soft":
            ### todo: see img sampling trainer 
            pass

    def sampler_regenerate_loss(self):
        # todo:
        pass


    