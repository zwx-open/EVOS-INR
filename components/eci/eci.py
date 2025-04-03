### todo: 更改用语
import torch
import numpy as np
from .scheduler import mt_scheduler_factory
from .laplacian import compute_laplacian_loss as compute_laplacian
import torch.nn.functional as F


class ECI(object):
    def __init__(
        self,
        input_img,
        device,
        use_ratio=0.5,
        num_epochs=5000,
        profile_interval_method="lin_dec",
        init_interval=75,
        end_interval=50,
        lap_coff=1e5,
        sample_num_schedular="constant",
        mutation_method="exp",
        init_mutation_ratio=0.6,
        end_mutation_ratio=0.4,
        crossover_method="add", # "no"
        profile_guide="value",
        use_laplace_epoch=1500,
        transform = None,
    ):
        self.input_img = input_img # chw
        self.C, self.H, self.W = self.input_img.shape
        self.device = device
        self.sample_num = self.H * self.W
        self.use_ratio = use_ratio
        self.num_epochs = num_epochs
        self.profile_interval_method = profile_interval_method
        self.init_interval = init_interval
        self.end_interval = end_interval
        self.lap_coff = lap_coff
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.init_mutation_ratio = init_mutation_ratio
        self.end_mutation_ratio = end_mutation_ratio
        self.crossover_method = crossover_method
        self.profile_guide = profile_guide
        self.use_laplace_epoch = use_laplace_epoch

        self.transform = transform

        self.cur_use_ratio = use_ratio

        self.use_ratio_scheduler = mt_scheduler_factory(sample_num_schedular)
        self.should_cache_lap = self.lap_coff > 0 or self.crossover_method != "no"
        if self.should_cache_lap:
            self.cached_gt_lap = compute_laplacian(self.input_img).squeeze()
        
        self.book = {}

    def _is_profile_freeze(self, epoch):
        _cur_interval = self._get_cur_interval(epoch)
        return epoch % _cur_interval == 1

    def _get_cur_interval(self, epoch):
        if self.profile_interval_method == "fixed":
            return self.init_interval
        elif self.profile_interval_method == "lin_dec":
            _start = self.init_interval
            _end = self.end_interval
            _cur_interval = _start + ((_end - _start) / self.num_epochs) * epoch
            return int(_cur_interval)

    def _get_cur_use_ratio(self, epoch):
        return self.use_ratio_scheduler(
            epoch, self.num_epochs, self.use_ratio
        )
    
    def _get_mutation_ratio(self, epoch):
        if self.mutation_method == "constant":
            return self.init_mutation_ratio * self.use_ratio
        elif self.mutation_method == "linear":
            _start = self.init_mutation_ratio
            _end = self.end_mutation_ratio  # max = 1
            ratio = _start + ((_end - _start) / self.num_epochs) * epoch
            return ratio * self.use_ratio
        elif self.mutation_method == "exp":
            _start = self.init_mutation_ratio
            _end = self.end_mutation_ratio
            _lamda = -np.log(_end / _start) / self.num_epochs
            ratio = _start * np.exp(-_lamda * epoch)
            return ratio * self.use_ratio
        else:
            raise NotImplementedError

    def _get_freeze_mask(self, epoch):
        mutation_ratio = self._get_mutation_ratio(epoch)  # constant
        _mask = torch.zeros(self.sample_num, dtype=torch.bool, device=self.device)
        cur_use_ratio = self.cur_use_ratio
        
        freeze_ratio = 1 - cur_use_ratio + mutation_ratio
        sorted_map_index = self.book["sorted_map_index"]

        freezed_num = int(freeze_ratio * self.sample_num)
        after_mutation_num = int((freeze_ratio - mutation_ratio) * self.sample_num)
        

        freezed_index = sorted_map_index[:freezed_num]

        # random
        sampled_index = torch.randperm(freezed_num, device=self.device)[
            :after_mutation_num
        ]
        after_melted_index = freezed_index[sampled_index]

        _mask[after_melted_index] = True
        self.book["freeze_mask"] = _mask
        return _mask

    def _update_freeze_info(self, pred, gt):
        error_map = F.mse_loss(pred, gt, reduction="none").mean(1)

        if self.crossover_method == "add":
            r_img = self._reconstruct_img(pred)
            # laplace_map = compute_laplacian(r_img, self.input_img).squeeze()
            laplace_map = F.mse_loss(compute_laplacian(r_img).squeeze(), self.cached_gt_lap, reduction="none")
            cross_lap_coff = self.lap_coff if self.lap_coff > 0 else 1e-5
            error_map = error_map + cross_lap_coff * laplace_map.flatten()
        
        elif self.crossover_method == "no":
            pass

        ### 直接用value还是一阶diff做guidance
        if self.profile_guide == "value":
            sorted_map_index = torch.argsort(error_map.flatten())
        # deprecated
        elif self.profile_guide == "diff_1":
            last_error_map = self.book.get("error_map", None)
            if last_error_map is None:
                last_error_map = torch.zeros_like(error_map)
            guidance_map = torch.abs(error_map - last_error_map)
            sorted_map_index = torch.argsort(guidance_map.flatten())
        else:
            raise NotImplementedError

        self.book["freeze_profile_pred"] = pred.detach()
        self.book["error_map"] = error_map.detach()
        self.book["sorted_map_index"] = sorted_map_index

    def _reconstruct_img(self, data) -> torch.tensor:
        img = data.reshape(self.H, self.W, self.C).permute(2, 0, 1)  
        img = self._decode_img(img)
        return img
    
    def _decode_img(self, img):
        # todo: customize here (vanilla: (-1,1) → (0,1) → (0,255))
        if self.transform is not None:
            data = self.transform.inverse(data)
        data = data * 255.0
        data = torch.clamp(data, min=0, max=255)
        return data
        return img

    def select_sample(self, coords, gt, epoch):
        if self._is_profile_freeze(epoch):
            return coords, gt
        else:
            freeze_mask = self._get_freeze_mask(epoch)
            _coords = coords[~freeze_mask]
            _gt = gt[~freeze_mask]

            return _coords, _gt

    def get_loss(self, pred, gt, epoch):
        # update freeze info
        if self._is_profile_freeze(epoch):
            self._update_freeze_info(pred, gt)
        
        mse = F.mse_loss(pred, gt)
        if self.lap_coff <= 0 or epoch > self.use_laplace_epoch:
            return mse
        else:
            profile_pred = self.book["freeze_profile_pred"]
            _mask = self.book["freeze_mask"]
            pseudo_full_pred = profile_pred.clone()
            # torch.cuda.synchronize()
            # log.pause_timer("lap_0")      
            # log.start_timer("lap_1")
            indices = torch.arange(_mask.shape[0], device=pred.device)[~_mask]
            pseudo_full_pred[indices] = pred
            # torch.cuda.synchronize()
            # log.pause_timer("lap_1")      
            # log.start_timer("lap_2")
            r_img = self._reconstruct_img(pseudo_full_pred)
            lap_loss = (
                # compute_laplacian(r_img, self.input_img)
                # .squeeze()
                F.mse_loss(compute_laplacian(r_img).squeeze(), self.cached_gt_lap, reduction="none")
                .flatten()[~_mask]
                .mean()
            )
            # torch.cuda.synchronize()
            # log.pause_timer("lap_2")      
            return mse + self.lap_coff * lap_loss

        
        