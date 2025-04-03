import cv2 as cv
import numpy as np
import torch

"""Implementation According to EGRA-NeRF: Edge-Guided Ray Allocation for Neural Radiance Fields"""
class EGRA(object):
    def __init__(self, img):
        self.c, self.h, self.w = img.shape
        self.img = img
        self.device = self.img.device
        self.img_gray_np = cv.cvtColor(
            img.permute(1, 2, 0).cpu().numpy(), cv.COLOR_RGB2GRAY
        )  # h,w
        self.T1 = 30
        self.T2 = 40
        self.p = 0.0025
        self.random_radius_1 = 6
        self.random_radius_2 = 9
        self.nerf_bsz = 4096
        self.nerf_N = 800 * 800  # nerf-syn
        self.edge_map = self._calc_edge_by_canny(self.img_gray_np)
        self.prob_map = self.edge_map * self.p + (1 - self.p) * (
            self.nerf_bsz / self.nerf_N
        )

        # normalization
        self.prob_map = self.prob_map / self.prob_map.sum()
        self.prob_map = self.prob_map.flatten()

    def _calc_edge_by_canny(self, img_gray_np):
        _map = cv.Canny(img_gray_np, self.T1, self.T2) / 255
        _map = torch.tensor(_map, device=self.device)
        return _map

    def _recalc_prob_map(self):
        # Dynamic Range
        edge_map_fla = self.edge_map.flatten()
        edge_map_indcies = torch.where(edge_map_fla == 1)[0]
        random_tensor = torch.randint(
            low=self.random_radius_1,
            high=self.random_radius_2 + 1,
            size=edge_map_indcies.shape,
            device=self.device,
        )
        edge_map_indcies = (edge_map_indcies + random_tensor) % (self.h * self.w)
        new_edge_map_fla = torch.zeros_like(edge_map_fla)
        new_edge_map_fla[edge_map_indcies] = 1

        prob_map = new_edge_map_fla * self.p + (1 - self.p) * (
            self.nerf_bsz / self.nerf_N
        )
        prob_map = prob_map / prob_map.sum()

        self.prob_map = prob_map

        return prob_map

    def sample(self, use_ratio):
        # cur_prob_map = self.prob_map
        cur_prob_map = self._recalc_prob_map()
        sampled_num = int(use_ratio * self.h * self.w)
        sampled_indicise = torch.multinomial(
            cur_prob_map, num_samples=sampled_num, replacement=False
        )
        return sampled_indicise
    
    def sample_by_num(self, sample_num):
        cur_prob_map = self._recalc_prob_map()
        sampled_indicise = torch.multinomial(
            cur_prob_map, num_samples=sample_num, replacement=False
        )
        return sampled_indicise

