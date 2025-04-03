import torch
import numpy as np

from .laplacian import compute_laplacian_loss as compute_laplacian
import torch.nn.functional as F


class EVOS(object):
    def __init__(
        self,
        input_data=None,
        device="cuda",
        signal_type="image",
        use_ratio=0.5,
        num_epochs=5000,
        fitness_interval_method="lin_dec",
        init_interval=100,
        end_interval=50,
        low_freq_lamda=1,
        high_freq_lamda=1e5,
        mutation_method="constant",
        init_mutation_ratio=0.5,
        end_mutation_ratio=0.5,
        crossover_method="add",  # "no" "select"
        use_cfs_epoch=1500,
    ):
        self.input_data = input_data  # chw
        self.signal_type = signal_type
        self.device = device
        self.num_epochs = num_epochs

        self.fitness_interval_method = fitness_interval_method
        self.init_interval = init_interval
        self.end_interval = end_interval

        self.low_freq_lamda = low_freq_lamda
        self.high_freq_lamda = high_freq_lamda
        self.crossover_method = crossover_method
        self.use_cfs_epoch = use_cfs_epoch

        self.mutation_method = mutation_method
        self.init_mutation_ratio = init_mutation_ratio
        self.end_mutation_ratio = end_mutation_ratio

        self.cur_use_ratio = use_ratio
        

        self.book = {}

        if self.signal_type == "image":
            self.C, self.H, self.W = self.input_data.shape
            self.cached_gt_lap = compute_laplacian(self.input_data).squeeze()
            self.should_cache_lap = self.lap_coff > 0 or self.crossover_method != "no"
            if self.should_cache_lap:
                self.cached_gt_lap = compute_laplacian(self.input_data).squeeze()

    def _is_fitness_eval_iter(self, epoch):
        _cur_interval = self._get_cur_fitness_interval(epoch)
        return epoch % _cur_interval == 1

    def _get_cur_fitness_interval(self, epoch):
        if self.fitness_interval_method == "fixed":
            return self.init_interval
        elif self.fitness_interval_method == "lin_dec":
            _start = self.init_interval
            _end = self.end_interval
            _cur_interval = _start + ((_end - _start) / self.num_epochs) * epoch
            return int(_cur_interval)

    def _get_mutation_ratio(self, epoch):
        if self.mutation_method == "constant":
            return self.init_mutation_ratio * self.cur_use_ratio
        elif self.mutation_method == "linear":
            _start = self.init_mutation_ratio
            _end = self.end_mutation_ratio  # max = 1
            ratio = _start + ((_end - _start) / self.num_epochs) * epoch
            return ratio * self.cur_use_ratio
        elif self.mutation_method == "exp":
            _start = self.init_mutation_ratio
            _end = self.end_mutation_ratio
            _lamda = -np.log(_end / _start) / self.num_epochs
            ratio = _start * np.exp(-_lamda * epoch)
            return ratio * self.cur_use_ratio
        else:
            raise NotImplementedError

    def _get_selection_mask(self, epoch):
        mutation_ratio = self._get_mutation_ratio(epoch)
        first_select_ratio = self.cur_use_ratio - mutation_ratio
        first_select_num = int(first_select_ratio * self.sample_num)

        sorted_map_index = self.book["sorted_map_index"]
        first_select_indices = sorted_map_index[-first_select_num:]

        # mutation
        mutation_num = int(mutation_ratio * self.sample_num)
        remain_indices = sorted_map_index[:-first_select_num]
        sample_index = torch.randperm(remain_indices.shape[0], device=self.device)[
            :mutation_num
        ]
        mutation_indicies = remain_indices[sample_index]

        selected_indices = torch.cat([first_select_indices, mutation_indicies])
        selection_mask = torch.zeros(
            self.sample_num, dtype=torch.bool, device=self.device
        )
        selection_mask[selected_indices] = True
        self.book["selecton_mask"] = selection_mask
        return selection_mask

    def _get_crossover_high_freq_map(self, pred, gt, recons_func):
        if self.signal_type == "image":
            # todo: 未完成重构测试
            r_img = recons_func(pred)
            laplace_map = F.mse_loss(
                compute_laplacian(r_img).squeeze(), self.cached_gt_lap, reduction="none"
            )
            return laplace_map.flatten()
        return None

    def crossover(self, pred, gt, epoch, recons_func):
        error_map = F.mse_loss(pred, gt, reduction="none").mean(1)
        high_freq_map = self._get_crossover_high_freq_map(pred, gt, recons_func)
        if not high_freq_map:
            high_freq_map = error_map

        if self.crossover_method == "no":
            sorted_map_index = torch.argsort(error_map.flatten())
        elif self.crossover_method == "add":
            hybrid_map = (
                self.low_freq_lamda * error_map + self.high_freq_lamda * high_freq_map
            )
            sorted_map_index = torch.argsort(hybrid_map.flatten())
        elif self.crossover_method == "select":
            # todo: 未完成重构测试
            sorted_low_freq_index = torch.argsort(error_map.flatten())
            sorted_high_freq_index = torch.argsort(high_freq_map.flatten())

            mutation_ratio = self._get_mutation_ratio(epoch)
            select_ratio = self.cur_use_ratio - mutation_ratio
            selected_num = int(self.sample_num * select_ratio)

            low_selected_index = sorted_low_freq_index[-selected_num:]
            high_selected_index = sorted_high_freq_index[-selected_num:]
            isin = torch.isin(low_selected_index, high_selected_index)
            selected_index = low_selected_index[isin] # intersection

            remain_num = selected_num - selected_index.shape[0]
            low_remain_index = low_selected_index[~isin]
            isin2 = torch.isin(high_selected_index, low_selected_index)
            high_remain_index = high_selected_index[~isin2]

            low_remain_num = int(
                remain_num
                * (
                    self.low_freq_lamda
                    * error_map.mean()
                    / (
                        self.high_freq_lamda * high_selected_index.mean()
                        + self.low_freq_lamda * error_map.mean()
                    )
                )
            )
            low_remain_num = min(low_remain_num, low_remain_index.shape[0])
            high_remain_num = remain_num - low_remain_num
            all_selected_index = torch.cat(
                [
                    high_remain_index[-high_remain_num:],
                    low_remain_index[-low_remain_num:],
                    selected_index,
                ]
            )
            all_remain_index = sorted_low_freq_index[
                ~torch.isin(sorted_low_freq_index, all_selected_index)
            ]
            # resort
            sorted_map_index = torch.cat([all_remain_index, all_selected_index])

        self.book["sorted_map_index"] = sorted_map_index
        self.book["fitness_eval_pred"] = pred.detach()

    def select_sample(self, coords, gt, epoch, use_ratio=0.5):
        self.sample_num = coords.shape[0]
        self.cur_use_ratio = use_ratio  # update - -
        if self._is_fitness_eval_iter(epoch):
            return coords, gt
        else:
            selection_mask = self._get_selection_mask(epoch)
            _coords = coords[selection_mask]
            _gt = gt[selection_mask]
            return _coords, _gt
    
    ############# via INDEX ###############
    def select_sample_via_index(self, coords, gt, epoch, use_ratio, index, full_coords, full_gt):
        '''shape ... / fitness全量推理 然后从采样的idx里面排'''
        self.sample_num = coords.shape[0]
        self.cur_use_ratio = use_ratio  # update - -
        if self._is_fitness_eval_iter(epoch):
            return full_coords, full_gt
        else:
            selection_mask = self._get_selection_mask_via_index(epoch, index)
            _coords = coords[selection_mask]
            _gt = gt[selection_mask]
            return _coords, _gt
    
    def _get_selection_mask_via_index(self, epoch, index):
        mutation_ratio = self._get_mutation_ratio(epoch)
        first_select_ratio = self.cur_use_ratio - mutation_ratio
        first_select_num = int(first_select_ratio * self.sample_num)

        sorted_map_index = self.book["sorted_map_index"]
        
        ###### via index ###########
        isin_ = torch.isin(sorted_map_index,index) # index可以有重复的
        cur_sorted_map_inedx = sorted_map_index[isin_]
        first_select_num = min(first_select_num, cur_sorted_map_inedx.shape[0])
        first_select_indices = cur_sorted_map_inedx[-first_select_num:]
        ###### via index ###########
    
        # mutation
        mutation_num = int(mutation_ratio * self.sample_num)
        remain_indices = cur_sorted_map_inedx[:-first_select_num]
        sample_index = torch.randperm(remain_indices.shape[0], device=self.device)[
            :mutation_num
        ]
        mutation_indicies = remain_indices[sample_index]

        selected_indices = torch.cat([first_select_indices, mutation_indicies])
        
        #  locate to index
        isin_ = torch.isin(index, selected_indices)
        selected_i = torch.nonzero(isin_).squeeze(1) 
        # selected_i.shape[0] >= first_select_num + mutation_num (因为重复)
        target_len =  first_select_num + mutation_num

        # Direct
        selected_i = selected_i[:target_len]
        
        # Uniform sampling
        # uniform_j =  torch.randperm(selected_i.shape[0], device=self.device)[
        #     :target_len
        # ]
        # selected_i = selected_i[uniform_j]
        
        selection_mask = torch.zeros(
            self.sample_num, dtype=torch.bool, device=self.device
        )
        selection_mask[selected_i] = True
        self.book["selecton_mask"] = selection_mask

        
        # @debug random selection mask
        # indices = torch.randperm(self.sample_num, device=self.device)[
        #             :target_len
        # ]
        # selection_mask[indices] = True
        # self.book["selecton_mask"] = selection_mask
        
        return selection_mask
    ############# via INDEX ###############

    def compute_cfs_loss(self, pred, gt, epoch, recons_func):
        # todo: 未完成重构测试
        '''Cross Supervision_loss'''
        mse = F.mse_loss(pred, gt)
        if epoch > self.use_cfs_epoch:
            return mse
        
        if self._is_fitness_eval_iter(epoch):
            pseudo_full_pred = pred
        else:
            fitness_eval_pred = self.book["fitness_eval_pred"]
            selecton_mask = self.book["selecton_mask"]
            pseudo_full_pred = fitness_eval_pred.clone()
            indices = torch.arange(selecton_mask.shape[0], device=pred.device)[selecton_mask]
            pseudo_full_pred[indices] = pred

        if self.signal_type == "image":
            r_img = recons_func(pseudo_full_pred)
            lap_loss = (
                        F.mse_loss(
                            compute_laplacian(r_img).squeeze(),
                            self.cached_gt_lap,
                            reduction="none",
                        )
                        .flatten()[selecton_mask]
                        .mean()
                    )
            return self.low_freq_lamda * mse + self.high_freq_lamda * lap_loss
        
        else:
            return mse

