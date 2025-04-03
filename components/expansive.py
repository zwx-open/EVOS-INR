import cv2 as cv
import numpy as np
import torch

class PriorGenerator(object):
    def __init__(self):
        pass

    def gen_prior_from_path(self, img_path, ratio):
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        edge = self._gen_prior_from_data(img, ratio)
        return edge

    def _gen_prior_from_data(self, img: np.ndarray, ratio=0.15, max_step=100, mu=15):
        thresh = [ratio * 0.8, ratio * 1.2]
        anchor, _ = cv.threshold(img, thresh=0, maxval=255, type=cv.THRESH_OTSU)
        thetas = [0.5, 1]
        edge = cv.Canny(img, anchor * thetas[0], anchor * thetas[1])
        step = 0
        sum_ = PriorGenerator.calc_edge_area(edge)

        while step < max_step:
            if PriorGenerator.in_thresh(sum_, thresh):
                break
            diff = sum_ - ratio
            thetas[1] *= 1 + diff * mu
            edge = cv.Canny(img, anchor * thetas[0], anchor * thetas[1])
            sum_ = PriorGenerator.calc_edge_area(edge)
            step += 1
        # print('sum_: ', sum_)
        
        return edge

    def gen_dilated_edge(self, edge):
        kernel_size = (2, 2)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
        edge = cv.dilate(edge, kernel, iterations=1)
        return edge

    @staticmethod
    def save_img(img, path):
        cv.imwrite(path, img)
        
    @staticmethod
    def calc_edge_area(edge: np.ndarray) -> float:
        # assert len(np.unique(edge)) == 2 and edge.max() == 255.0
        area_perc = edge.sum() / (edge.size * 255.0)
        assert area_perc >= 0 and area_perc <= 1
        return area_perc

    @staticmethod
    def in_thresh(x, thresh):
        assert len(thresh) == 2
        return x <= thresh[1] and x >= thresh[0]


class ExpansiveSupervision(object):
    def __init__(self, img, anchor_ratio=0.2):
        self.prior_gen = PriorGenerator()
        self.anchor_ratio = anchor_ratio
        self.c, self.h, self.w = img.shape
        self.img = img
        self.device = self.img.device
        self.img_gray_np = cv.cvtColor(img.permute(1,2,0).cpu().numpy(), cv.COLOR_RGB2GRAY)  # h,w
        self.prior = self._init_prior()
        self.prior = torch.tensor(self.prior, device = self.device).to(torch.float32)
    
    def _init_prior(self):
        prior = self.prior_gen._gen_prior_from_data(self.img_gray_np, ratio = self.anchor_ratio)
        prior = self.prior_gen.gen_dilated_edge(prior)
        prior_perc = self.prior_gen.calc_edge_area(prior)
        if prior_perc > self.anchor_ratio:
            indexes = np.argwhere(prior > 0)
            cali_len = int(self.anchor_ratio * self.w * self.h)
            drop_len = len(indexes) - cali_len
            to_drop = np.random.choice(np.arange(len(indexes)), size=drop_len, replace=False)
            to_drop_indexes = indexes[to_drop]
            prior[to_drop_indexes[:,0],to_drop_indexes[:,1]] = 0
        prior_perc = self.prior_gen.calc_edge_area(prior)
        prior = prior / 255.0
        return prior


    def select_sample(self, use_ratio):
        # 注意devices

        source_ratio = use_ratio - self.anchor_ratio
        source_num = int(source_ratio * self.h * self.w)
        anchor_indices = torch.where(self.prior.flatten() == 1)[0]
        other_indices = torch.where(self.prior.flatten() == 0)[0]

        # random source
        shuffled_indices = torch.randperm(other_indices.shape[0], device=self.device)[
            :source_num
        ]
        source_indices = other_indices[shuffled_indices]
        indices = torch.concat((anchor_indices, source_indices))
    
        return indices