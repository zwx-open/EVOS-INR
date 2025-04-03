from models import Siren, PEMLP
from util.logger import log
from util.tensorboard import writer
from util.plotter import plotter
from util.misc import fix_seed, calc_intersec
from util import io

from trainer.img_trainer import ImageTrainer

from components.ssim import compute_ssim_loss as compute_ssim
from components.laplacian import compute_laplacian_loss as compute_laplacian
from components.lmc import LMC
from components.nmt import NMT
from components.nmt import mt_scheduler_factory
from components.expansive import ExpansiveSupervision as ES
from components.egra import EGRA
from components.eci import ECI

import numpy as np
import imageio.v2 as imageio
import torch
import torch.nn.functional as F

from tqdm import trange
import json


class ImageSamplingTrainer(ImageTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.use_grad_scaler = False  # used in soft mining origin paper | 效果差不多
        self.book = {}
        self.use_ratio_scheduler = mt_scheduler_factory(self.args.sample_num_schedular)
        self.should_cache_lap = (
            self.args.lap_coff > 0 or self.args.crossover_method != "no"
        )
        self.record_indices = {}  # for recording



    def _nmt_init(self):
        self.nmt = NMT(
            self.model,
            self.args.num_epochs,
            (self.H, self.W, self.C),
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

    def _es_init(self):
        self.es = ES(img=self.input_img, anchor_ratio=0.5 * self.args.use_ratio)

    def _egra_init(self):
        self.egra = EGRA(img=self.input_img)

    def _eci_init(self):
        self.eci = ECI(
            input_img=self.input_img,
            device=self.input_img.device,
            use_ratio=self.args.use_ratio,
            num_epochs=self.args.num_epochs,
            profile_interval_method=self.args.profile_interval_method,
            init_interval=self.args.init_interval,
            end_interval=self.args.end_interval,
            lap_coff=self.args.lap_coff,
            sample_num_schedular=self.args.sample_num_schedular,
            mutation_method=self.args.mutation_method,
            init_mutation_ratio=self.args.init_mutation_ratio,
            end_mutation_ratio=self.args.end_mutation_ratio,
            crossover_method=self.args.crossover_method,
            profile_guide=self.args.profile_guide,
            use_laplace_epoch=self.args.use_laplace_epoch,
            transform=self.transform,
        )

    def _reset_rng(self):
        generator = torch.Generator()
        seed = generator.seed()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _recover_rng(self):
        fix_seed(self.args.seed)

    def _get_inference_samples(self, epoch):
        coords, gt = self.full_coords, self.full_gt
        self._reset_rng()
        _st = self.args.strategy
        cur_use_ratio = self.cur_use_ratio

        if epoch <= self.args.warm_up:
            return coords, gt

        if _st == "full":
            return coords, gt

        elif _st == "random":
            _ratio_len = int(self.sample_num * cur_use_ratio)
            indices = torch.randperm(self.sample_num, device=self.device)[:_ratio_len]
            random_coords = self.full_coords[indices]
            random_gt = self.full_gt[indices]
            self.book["cur_sampled_indices"] = indices
            return random_coords, random_gt

        elif _st == "freeze":
            if self._is_profile_freeze(epoch):
                return coords, gt
            else:
                # freeze_mask = self._get_freeze_mask(epoch)
                # _coords = self.full_coords[~freeze_mask]
                # _gt = self.full_gt[~freeze_mask]
                selection_mask = self._get_selection_mask(epoch)
                _coords = self.full_coords[selection_mask]
                _gt = self.full_gt[selection_mask]
                return _coords, _gt

        elif _st == "eci":
            sampled_coords, sampled_gt = self.eci.select_sample(coords, gt, epoch)
            return sampled_coords, sampled_gt

        elif _st == "nmt":
            sampled_coords, sampled_gt, _, indices = self.nmt.sample(
                epoch - 1, coords, gt
            )
            # record indices
            self.book["cur_sampled_indices"] = indices
            return sampled_coords, sampled_gt

        elif _st == "soft":
            net_grad = self.book.get("soft_net_grad", None)
            loss_per_pix = self.book.get("soft_loss_per_pix", None)

            points_2d = self.lmc(net_grad, loss_per_pix)
            if self.args.soft_raw:
                points_2d.requires_grad = True

            x, y = points_2d[:, 1], points_2d[:, 0]
            indices = (self.H * y + x).to(torch.long)
            self.book["cur_sampled_indices"] = indices

            if self.args.soft_raw:
                x_ = (x / self.W) * 2 - 1
                y_ = (y / self.H) * 2 - 1
                _coords = torch.stack((y_, x_), dim=1)
            else:
                _coords = self.full_coords[indices]

            _gt = self.full_gt[indices]

            self.book["soft_points_2d"] = points_2d


            return _coords, _gt

        elif _st == "expansive":
            indices = self.es.select_sample(use_ratio=cur_use_ratio)
            self.book["cur_sampled_indices"] = indices
            _coords = self.full_coords[indices]
            _gt = self.full_gt[indices]

            

            return _coords, _gt

        elif _st == "egra":
            indices = self.egra.sample(use_ratio=cur_use_ratio)
            self.book["cur_sampled_indices"] = indices
            _coords = self.full_coords[indices]
            _gt = self.full_gt[indices]


            return _coords, _gt

        raise NotImplementedError

    def _compute_sample_loss(self, pred, gt, epoch):
        _st = self.args.strategy
        mse = self.compute_mse(pred, gt)
        # standard_loss = mse
        # if self.args.lap_cpff > 0:
        if _st == "full":
            return self._compose_loss(mse, pred, gt, epoch)
        elif _st == "random":
            return mse
        elif _st == "freeze":
            if self._is_profile_freeze(epoch) or epoch <= self.args.warm_up:
                return self._compose_loss(mse, pred, gt, epoch)
            else:
                if self.args.lap_coff <= 0 or epoch > self.args.use_laplace_epoch:
                    return mse
                else:
                    # log.start_timer("lap_0")
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
                    r_img = self.reconstruct_img(pseudo_full_pred)
                    lap_loss = (
                        # compute_laplacian(r_img, self.input_img)
                        # .squeeze()
                        F.mse_loss(
                            compute_laplacian(r_img).squeeze(),
                            self.cached_gt_lap,
                            reduction="none",
                        )
                        .flatten()[~_mask]
                        .mean()
                    )
                    # torch.cuda.synchronize()
                    # log.pause_timer("lap_2")

                    return mse + self.args.lap_coff * lap_loss

                # return mse  # 暂时先用这个
                # recompute freeze | 目前先不管结构信息(我们的方法能用高阶loss做引导)
                return _mse

        elif _st == "soft":
            loss_per_pix = self.book["soft_loss_per_pix"]
            loss = loss_per_pix.mean()
            if self.use_grad_scaler:
                loss = self.grad_scaler.scale(loss)
            return loss
        else:
            return mse

    def _compose_loss(self, cur_loss, pred, gt, epoch):
        if self.args.lap_coff > 0:
            r_img = self.reconstruct_img(pred)
            lap_loss = F.mse_loss(
                compute_laplacian(r_img).squeeze(), self.cached_gt_lap
            )

            cur_loss += self.args.lap_coff * lap_loss
        return cur_loss

    def _get_mutation_ratio(self, epoch):
        if self.args.mutation_method == "constant":
            return self.args.init_mutation_ratio * self.args.use_ratio
        elif self.args.mutation_method == "linear":
            # 0.5 | 0.4 → 0.6
            _start = self.args.init_mutation_ratio
            _end = self.args.end_mutation_ratio  # max = 1
            ratio = _start + ((_end - _start) / self.args.num_epochs) * epoch
            return ratio * self.args.use_ratio
        elif self.args.mutation_method == "exp":
            _start = self.args.init_mutation_ratio
            _end = self.args.end_mutation_ratio
            _lamda = -np.log(_end / _start) / self.args.num_epochs
            ratio = _start * np.exp(-_lamda * epoch)
            return ratio * self.args.use_ratio
        else:
            raise NotImplementedError
    
    def _get_selection_mask(self,epoch):
        mutation_ratio = self._get_mutation_ratio(epoch) 
        first_select_ratio = self.cur_use_ratio - mutation_ratio
        first_select_num = int(first_select_ratio * self.sample_num)
        
        sorted_map_index = self.book["sorted_map_index"]
        first_select_indices= sorted_map_index[-first_select_num:]
        
        # mutation
        mutation_num = int(mutation_ratio * self.sample_num)
        remain_indices = sorted_map_index[:-first_select_num]
        sample_index = torch.randperm(remain_indices.shape[0], device=self.device)[:mutation_num]
        mutation_indicies = remain_indices[sample_index]

        selected_indices = torch.cat([first_select_indices, mutation_indicies])
        self.book["cur_sampled_indices"] = selected_indices
        
        _mask = torch.ones(self.sample_num, dtype=torch.bool, device=self.device)
        _mask[selected_indices] = False
        self.book["freeze_mask"] = _mask

        selection_mask = torch.zeros(self.sample_num, dtype=torch.bool, device=self.device)
        selection_mask[selected_indices] = True
        return selection_mask

    def _get_freeze_mask(self, epoch):
        ### deprecated
        mutation_ratio = self._get_mutation_ratio(epoch) 
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
        self.book["cur_sampled_indices"] = after_melted_index  ## inverse...
        return _mask

    def _is_profile_freeze(self, epoch):
        if self.args.strategy != "freeze":
            return False
        # if epoch <=  self.args.warm_up:
        #     return False
        _cur_interval = self._get_cur_interval(epoch)
        return epoch % _cur_interval == 1

    def _get_cur_interval(self, epoch):
        if self.args.profile_interval_method == "fixed":
            return self.args.init_interval
        elif self.args.profile_interval_method == "lin_dec":
            _cur_ratio = self.cur_use_ratio
            _start = self.args.init_interval
            _end = self.args.end_interval
            _cur_interval = _start + ((_end - _start) / self.args.num_epochs) * epoch
            return int(_cur_interval)

    def _get_cur_use_ratio(self, epoch):
        return self.use_ratio_scheduler(
            epoch, self.args.num_epochs, self.args.use_ratio
        )

    def _update_freeze_info(self, pred, gt, epoch):
        error_map = F.mse_loss(pred, gt, reduction="none").mean(1)
        if self.args.crossover_method == "add":
            r_img = self.reconstruct_img(pred)
            # laplace_map = compute_laplacian(r_img, self.input_img).squeeze()
            laplace_map = F.mse_loss(
                compute_laplacian(r_img).squeeze(), self.cached_gt_lap, reduction="none"
            )
            cross_lap_coff = self.args.lap_coff if self.args.lap_coff > 0 else 1e-5
            error_map = error_map + cross_lap_coff * laplace_map.flatten()
        elif self.args.crossover_method == "no":
            pass
        

        ### 直接用value还是一阶diff做guidance
        if self.args.profile_guide == "value":
            sorted_map_index = torch.argsort(error_map.flatten())
        # deprecated
        elif self.args.profile_guide == "diff_1":
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

        # 按照当前频率分量的比值选择是否crossover
        if self.args.crossover_method == "select":
            r_img = self.reconstruct_img(pred)
            # laplace_map = compute_laplacian(r_img, self.input_img).squeeze()
            laplace_map = F.mse_loss(
                compute_laplacian(r_img).squeeze(), self.cached_gt_lap, reduction="none"
            )
            cross_lap_coff = self.args.lap_coff if self.args.lap_coff > 0 else 1e-5
            laplace_error_map = cross_lap_coff * laplace_map.flatten()
            sorted_lap_map_index = torch.argsort(laplace_error_map.flatten())
            self.book["sorted_lap_map_index"] = sorted_lap_map_index

            mutation_ratio = self._get_mutation_ratio(epoch) 
            freeze_ratio = 1 - self.cur_use_ratio + mutation_ratio

            freezed_num = int(freeze_ratio * self.sample_num)
            selected_num = self.sample_num - freezed_num

            l2_error_selected_index = sorted_map_index[-selected_num:]
            lap_error_selected_index = sorted_lap_map_index[-selected_num:]
            isin = torch.isin(l2_error_selected_index, lap_error_selected_index)
            
            selected_index = l2_error_selected_index[isin]

            remain_num = selected_num - selected_index.shape[0]
            l2_remain_index = l2_error_selected_index[~isin]
            isin2 = torch.isin(lap_error_selected_index, l2_error_selected_index)
            lap_remain_index =  lap_error_selected_index[~isin2]

            l2_remain_num = int(remain_num * (error_map.mean() / (laplace_error_map.mean() +error_map.mean())))
            l2_remain_num = min(l2_remain_num, l2_remain_index.shape[0])
            lap_remain_num = remain_num - l2_remain_num
            all_selected_index = torch.cat([lap_remain_index[-lap_remain_num:],l2_remain_index[-l2_remain_num:], selected_index])
            all_remain_index = sorted_map_index[~torch.isin(sorted_map_index, all_selected_index)]
            select_sorted_index = torch.cat([all_remain_index, all_selected_index])
            self.book["sorted_map_index"] = select_sorted_index


    def train(self):
        num_epochs = self.args.num_epochs
        coords, gt = self._get_data()
        self.model = self._get_model(in_features=2, out_features=3).to(self.device)

        coords = coords.to(self.device)
        gt = gt.to(self.device)
        self.input_img = self.input_img.to(self.device)

        self.full_coords = coords
        self.full_gt = gt
        self.sample_num = coords.shape[0]

        optimizer = torch.optim.Adam(lr=self.args.lr, params=self.model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1 ** min(iter / num_epochs, 1)
        )
        self.grad_scaler = torch.cuda.amp.GradScaler(2**10)

        if self.should_cache_lap:
            self.cached_gt_lap = compute_laplacian(self.input_img).squeeze()

        if self.args.strategy == "nmt":
            self._nmt_init()

        elif self.args.strategy == "soft":
            self._soft_init()

        elif self.args.strategy == "expansive":
            self._es_init()

        elif self.args.strategy == "egra":
            self._egra_init()

        elif self.args.strategy == "eci":
            self._eci_init()

        for epoch in trange(1, num_epochs + 1):
            torch.cuda.synchronize()
            log.start_timer("train")

            log.start_timer("sampler")
            self.cur_use_ratio = self._get_cur_use_ratio(epoch)
            coords, gt = self._get_inference_samples(epoch)

            if self.args.strategy == "soft" and not self.args.soft_raw:
                coords.requires_grad = True

            torch.cuda.synchronize()
            log.pause_timer("sampler")

            self._recover_rng()

            log.start_timer("inference")
            pred = self.model(coords)

            torch.cuda.synchronize()
            log.pause_timer("inference")

            log.start_timer("sampler")
            if self._is_profile_freeze(epoch):
                log.start_timer("freeze_info")
                self._update_freeze_info(pred, gt, epoch)
                torch.cuda.synchronize()
                log.pause_timer("freeze_info")

            # soft mining logics
            if self.args.strategy == "soft":
                loss_per_pix = F.mse_loss(pred, gt, reduction="none").mean(-1)
                l1_map = torch.abs(pred - gt).mean(-1).detach()  # important_loss
                correction = (
                    1.0
                    / torch.clip(l1_map, min=torch.finfo(torch.float16).eps).detach()
                )
                alpha = self.args.soft_mining_alpha
                r = min((epoch / self.args.soft_mining_warmup), alpha)
                correction.pow_(r)
                correction.clamp_(min=0.2, max=correction.mean() + correction.std())

                if not self.args.wo_correction_loss:
                    loss_per_pix.mul_(correction)

                self.book["soft_loss_per_pix"] = loss_per_pix

            torch.cuda.synchronize()
            log.pause_timer("sampler")

            log.start_timer("compute_loss")
            loss = self._compute_sample_loss(pred, gt, epoch)
            # test
            if self.args.strategy == "eci":
                loss = self.eci.get_loss(pred, gt, epoch)
            torch.cuda.synchronize()
            log.pause_timer("compute_loss")

            log.start_timer("backward")
            optimizer.zero_grad()
            loss.backward()

           

            optimizer.step()
            if self.args.use_lr_scheduler:
                scheduler.step()

            # Get Grad → sampler
            if self.args.strategy == "soft":
                with torch.no_grad():

                    if self.args.soft_raw:
                        # cannot learn anymore:
                        net_grad = self.book["soft_points_2d"].grad.detach()
                    else:
                        # update
                        net_grad = coords.grad.detach()

                    scale_ = self.grad_scaler._scale if self.use_grad_scaler else 1
                    net_grad = net_grad / (
                        (scale_ * correction * loss_per_pix).unsqueeze(1)
                        + torch.finfo(net_grad.dtype).eps
                    )
                    self.book["soft_net_grad"] = net_grad

            torch.cuda.synchronize()
            log.pause_timer("backward")

            torch.cuda.synchronize()
            log.pause_timer("train")

            

            if epoch % self.args.log_epoch == 0:

                # torch.cuda.empty_cache()

                full_pred = self.model(self.full_coords)
                r_img = self.reconstruct_img(full_pred)

                psnr = self.compute_psnr(r_img, self.input_img).detach().item()
                mse = self.compute_mse(full_pred, self.full_gt).detach().item()
                ssim = compute_ssim(r_img, self.input_img).detach().item()

                if self.args.eval_lpips:
                    lpips = self.calc_lpips.compute_lpips(r_img, self.input_img)
                else:
                    lpips = -1

                self.recorder[epoch]["psnr"] = psnr
                self.recorder[epoch]["mse"] = mse
                self.recorder[epoch]["ssim"] = ssim
                self.recorder[epoch]["lpips"] = lpips

                log.inst.info(f"Epoch {epoch}: PSNR: {psnr}")
                log.inst.info(f"Epoch {epoch}: MSE: {mse}")
                log.inst.info(f"Epoch {epoch}: SSIM: {ssim}")
                log.inst.info(f"Epoch {epoch}: LPIPS: {lpips}")

                if self.args.strategy == "nmt":
                    log.inst.info(f"nmt_ratio_{self.nmt.get_ratio()}")
                    log.inst.info(f"nmt_interval_{self.nmt.get_interval()}")


            if epoch % self.args.snap_epoch == 0:
                _, final_img = self.inference()
                io.save_cv2(
                    final_img,
                    self._get_sub_path(
                        f"snap_images_epoch_{epoch}", f"{self.data_name}.png"
                    ),
                )

            if self.args.dense_measure_psnr:
                if not self._is_profile_freeze(epoch):
                    with torch.no_grad():
                        full_pred = self.model(self.full_coords)
                        r_img = self.reconstruct_img(full_pred)
                        psnr = self.compute_psnr(r_img, self.input_img).detach().item()
                        mse = self.compute_mse(full_pred, self.full_gt).detach().item()
                        ssim = compute_ssim(r_img, self.input_img).detach().item()

                        writer.inst.add_scalar(
                        f"{self.data_name}/train/dense_psnr",
                            psnr,
                            global_step=epoch,
                        )

                        writer.inst.add_scalar(
                        f"{self.data_name}/train/dense_ssim",
                            ssim,
                            global_step=epoch,
                        )

                

            self.book["pred"] = pred
            writer.inst.add_scalar(
                f"{self.data_name}/train/total_loss",
                loss.detach().item(),
                global_step=epoch,
            )

        _, final_img = self.inference()
        io.save_cv2(
            final_img, self._get_sub_path("final_pred", f"{self.data_name}.png")
        )

        self._save_ckpt(num_epochs, self.model, optimizer, scheduler)

    def inference(self):
        with torch.no_grad():
            pred = self.model(self.full_coords).cpu()

            r_img = self.reconstruct_img(pred).permute(1, 2, 0).numpy()  # h,w,c
        return pred, r_img



    


        
