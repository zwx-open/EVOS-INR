from models import Siren, PEMLP
from util.logger import log
from util.tensorboard import writer
from util import io

from trainer.base_trainer import BaseTrainer
from components.ssim import compute_ssim_loss
from components.lpips import Calc_LPIPS

import numpy as np
import imageio.v2 as imageio
import torch
import os
import torch.nn.functional as F

from tqdm import trange


class ImageTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self._parse_input_data()

        if self.args.eval_lpips:
            self.calc_lpips = Calc_LPIPS(device=self.device)

    def _parse_input_data(self):
        path = self.args.input_path
        img = torch.from_numpy(imageio.imread(path)).permute(2, 0, 1)  # c,h,w
        self.input_img = img
        self.gt = img
        self.C, self.H, self.W = img.shape

    def _encode_img(self, img):
        img = torch.clamp(img, min=0, max=255)
        img = img / 255.0
        img = self.transform.tranform(img)
        return img

    def _decode_img(self, data):
        data = self.transform.inverse(data)
        data = data * 255.0
        data = torch.clamp(data, min=0, max=255)
        return data

    def _get_data(self):
        img = self.input_img
        img = self._encode_img(img)

        # @test inverse
        # r_img = self._decode_img(img)
        # psnr_ = self.compute_psnr(r_img, self.input_img)
        # print('psnr_: ', psnr_)
        # exit()

        gt = img.permute(1, 2, 0).reshape(-1, self.C)  # h*w, C
        coords = torch.stack(
            torch.meshgrid(
                [torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W)],
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        return coords, gt

    def train(self):
        loss_list = []
        num_epochs = self.args.num_epochs
        coords, gt = self._get_data()
        model = self._get_model(in_features=2, out_features=3).to(self.device)

        coords = coords.to(self.device)
        gt = gt.to(self.device)
        self.input_img = self.input_img.to(self.device)

        optimizer = torch.optim.Adam(lr=self.args.lr, params=model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda iter: 0.1 ** min(iter / num_epochs, 1)
            )
        
        for epoch in trange(1, num_epochs + 1):
            log.start_timer("train")
            pred = model(coords)
            recons_pred = self.reconstruct_img(pred)
            mse = self.compute_mse(pred, gt)
            psnr = self.compute_psnr(recons_pred, self.input_img)
            
            loss = self.compute_mse(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            torch.cuda.synchronize()
            log.pause_timer("train")

            if epoch % self.args.log_epoch == 0:
                log.inst.info(f"epoch: {epoch} → loss {loss:.8f}")
                log.inst.info(f"epoch: {epoch} → psnr {psnr:.4f}")

                self.recorder[epoch]["psnr"] = psnr.detach().item()
                self.recorder[epoch]["mse"] = mse.detach().item()
                
                # ssim
                ssim = self.compute_ssim(recons_pred, self.input_img)
                self.recorder[epoch]["ssim"] = ssim.detach().item()

            writer.inst.add_scalar(
                f"{self.data_name}/train/total_loss",
                loss.detach().item(),
                global_step=epoch,
            )
            
            writer.inst.add_scalar(
                f"{self.data_name}/train/psnr", psnr.detach().item(), global_step=epoch
            )
            writer.inst.add_scalar(
                f"{self.data_name}/train/mse", mse.detach().item(), global_step=epoch
            )

            if len(loss_list) > 1:
                # record loss components
                for i, cur_loss in enumerate(loss_list):
                    writer.inst.add_scalar(
                        f"{self.data_name}/train/loss_{i}",
                        cur_loss.detach().item(),
                        global_step=epoch,
                    )

        with torch.no_grad():
            pred = model(coords).cpu()
            # pred distribute
            # plotter.plot_hist(
            #     pred, self._get_sub_path("pred_hist", f"{self.data_name}.png")
            # )

            final_img = (
                self.reconstruct_img(pred).permute(1, 2, 0).numpy()
            )  # h,w,c
            io.save_cv2(
                final_img, self._get_sub_path("final_pred", f"{self.data_name}.png")
            )

        self._save_ckpt(num_epochs, model, optimizer, scheduler)

    def reconstruct_img(self, data) -> torch.tensor:
        img = data.reshape(self.H, self.W, self.C).permute(2, 0, 1)  # c,h,w
        img = self._decode_img(img)
        return img

    @staticmethod
    def compute_psnr(pred, gt):
        """image data"""
        mse = BaseTrainer.compute_mse(pred, gt)
        return 20.0 * torch.log10(gt.max() / torch.sqrt(mse))

    @staticmethod
    def compute_ssim(pred, gt):
        return compute_ssim_loss(pred, gt)

    @staticmethod
    def compute_normalized_psnr(pred, gt):
        """normalized 0 - 1"""
        mse = BaseTrainer.compute_mse(pred, gt)
        return -10.0 * torch.log10(mse)
