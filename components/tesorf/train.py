import os
from tqdm.auto import tqdm
from util.logger import log
from components.lmc import LMC
from components.expansive import ExpansiveSupervision
from components.egra import EGRA
from .tensorf_opt import config_parser, merge_args


import json, random
from .renderer import *
from .utils import *
from torch.utils.tensorboard import SummaryWriter
from util.tensorboard import writer
from util.logger import log
from util.misc import fix_seed
import datetime

from .dataLoader import dataset_dict
import sys


renderer = OctreeRender_trilinear_fast


def _reset_randomness():
    generator = torch.Generator()
    seed = generator.seed()

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _reset_fixed_seed(seed):
    fix_seed(seed)



class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr : self.curr + self.batch]


class EgraSampler:
    def __init__(self, datasets, args):
        self.bs = args.batch_size
        self.args = args
        self.datasets = datasets
        self.device = args.device
        self.w, self.h = datasets.img_wh
        self.img_nums = len(datasets.meta["frames"])
        self.images = (datasets.all_rgbs.reshape(self.img_nums, self.h, self.w, -1) * 255.0).to(
            torch.uint8
        ).to(self.device)
        # easy implementation
        prob_maps = []
        for image in tqdm(self.images):
            egra = EGRA(
                image.permute(2, 0, 1) 
            )
            cur_prob_map = egra._recalc_prob_map()

            prob_maps += [cur_prob_map]
        
        self.prob_maps = torch.cat(prob_maps,0)
        self.prob_maps = self.prob_maps / self.prob_maps.sum()

    
    def nextids(self):
        ### tooo slow recalc to make it parallel

        # average_bs = np.ceil(self.bs / self.img_nums).astype(np.int32)
        # sample_ids = []
        # for i,egra in enumerate(self.erga_calculators):
        #     sampled_indices = egra.sample_by_num(average_bs)
        #     glob_sampled_indices = i * (self.h*self.w) + sampled_indices
        #     sample_ids += [glob_sampled_indices]
        # sample_ids = torch.cat(sample_ids, 0)
        
        # drop 4100-4096 = 4 ||| implement in soft mining
        # to_drop_num = (average_bs*self.img_nums) - self.bs
        # to_drop_indices = torch.multinomial(torch.ones(average_bs*self.img_nums), to_drop_num, replacement=False)
        # mask = torch.ones_like(sample_ids, dtype=torch.bool)
        # mask[to_drop_indices] = False
        # cali_sample_ids = sample_ids[mask]

        # return cali_sample_ids
        
        # less than 2**24 ||| still slow
        group = int(np.ceil(self.prob_maps.shape[0] / (2**24)))
        sample_ids = []
        group_len = int(np.ceil(self.prob_maps.shape[0] / group))
        sample_len = int(np.ceil(self.bs / group))
        for i in range(int(group)):
            _final = min((i+1)*group_len, self.prob_maps.shape[0])
            cur_prob_maps = self.prob_maps[i*group_len:_final]
            cur_prob_maps = cur_prob_maps  / cur_prob_maps.sum()
            cur_sampled_ids = torch.multinomial(
            cur_prob_maps ,sample_len , replacement=False
            )
            cur_sampled_ids = cur_sampled_ids + (i * group_len)
            sample_ids += [cur_sampled_ids]
        
        sample_ids = torch.cat(sample_ids, 0)

        ### drop
        if sample_ids.shape[0] > self.bs:
            to_drop_num = sample_ids.shape[0] - self.bs
            to_drop_indices = torch.multinomial(torch.ones(sample_ids.shape[0]), to_drop_num, replacement=False)
            mask = torch.ones_like(sample_ids, dtype=torch.bool)
            mask[to_drop_indices] = False
            sample_ids = sample_ids[mask]

        return sample_ids


class ExpansiveSupSampler:
    def __init__(self, datasets, args):
        self.bs = args.batch_size
        self.args = args
        self.datasets = datasets
        self.device = args.device
        self.alpha_a = 0.25
        self.beta_a = args.beta_a  # anchor points / all points
        self.w, self.h = datasets.img_wh
        self.img_nums = len(datasets.meta["frames"])
        self.images = (datasets.all_rgbs.reshape(self.img_nums, self.h, self.w, -1) * 255.0).to(
            torch.uint8
        )
        self.priors = []
        for image in tqdm(self.images):
            prior = ExpansiveSupervision(
                image.permute(2, 0, 1), anchor_ratio=self.alpha_a 
            ).prior  # c,h,w
            self.priors += [prior]
        self.priors = torch.stack(self.priors, 0)

        self.anchor_indices = torch.where(self.priors.flatten() == 1)[0]
        self.other_indices = torch.where(self.priors.flatten() == 0)[0]

        self.anchor_num = int(self.bs * self.beta_a)
        self.source_num = self.bs - self.anchor_num

        self.kamma = 1.0 # linear loss

        # ablation
        # self.beta_a = 1 # w/o source sup.
        # self.beta_a = 1 # w/o anchor sup.
        self.lamda_a = args.lamda_a # 1.
        self.lamda_s = args.lamda_s # 1.


    def nextids(self):
        # random permute tooo slow here | layer sample
        anchor_samples_indices = torch.randint(0, self.anchor_indices.shape[0], size=(self.anchor_num,), device=self.device)
        source_samples_indices = torch.randint(0, self.other_indices.shape[0], size=(self.source_num,), device=self.device)
        final_anchor_idx = self.anchor_indices[anchor_samples_indices]
        final_source_idx = self.other_indices[source_samples_indices]
        sample_idx = torch.concat((final_anchor_idx, final_source_idx))


        # shuffled_a_indices = torch.randperm(
        #     self.anchor_indices.shape[0], device=self.device
        # )[: self.anchor_num]
        # final_anchor_idx = self.anchor_indices[shuffled_a_indices]

        # shuffled_s_indices = torch.randperm(
        #     self.other_indices.shape[0], device=self.device
        # )[: self.source_num]
        # final_source_idx = self.other_indices[shuffled_s_indices]
        # sample_idx = torch.concat((final_anchor_idx, final_source_idx))

        return sample_idx
    
    def compute_loss(self, rgb_map, rgb_train, iteration):
        anchor_mse = ((rgb_map[:self.anchor_num,:] - rgb_train[:self.anchor_num,:]) ** 2).mean()
        source_mse = ((rgb_map[self.anchor_num:,:] - rgb_train[self.anchor_num:,:]) ** 2).mean()
        
        ## writer
        # lamda = anchor_mse / source_mse
        # writer.inst.add_scalar(
        #         "train/anchor_mse", anchor_mse.detach().item(), global_step=iteration
        #     )
        # writer.inst.add_scalar(
        #         "train/source_mse", source_mse.detach().item(), global_step=iteration
        #     )
        # writer.inst.add_scalar(
        #         "train/lamda", lamda.detach().item(), global_step=iteration
        #     )


        
        source_coff = ((1-self.alpha_a) / self.alpha_a) * self.kamma
        
        if self.args.a_s_add: # ablation
            cur_source_coff = 1.
        else:
            cur_source_coff = source_coff + (1-source_coff) * iteration / self.args.n_iters
        loss = self.lamda_a * anchor_mse + self.lamda_s * cur_source_coff * source_mse
       

        if self.args.use_mse:
            mse = ((rgb_map - rgb_train) ** 2).mean()
            return mse
        
        return loss


class SoftminingSampler:
    def __init__(self, datasets, args):
        self.args = args
        self.datasets = datasets
        self.device = args.device
        self.bs = args.batch_size
        w, h = datasets.img_wh
        img_nums = len(datasets.meta["frames"])
        self.images = (
            (datasets.all_rgbs.reshape(img_nums, h, w, -1) * 255.0)
            .to(torch.uint8)
            .to(self.device)
        )

        avearge_bs = np.ceil(self.bs / img_nums).astype(np.int32)
        cali_bs = avearge_bs * img_nums
        self.const_img_id = torch.arange(
            0, self.images.shape[0], device=self.device
        ).repeat_interleave(avearge_bs)

        self.lmc = LMC(
            images=self.images,
            num_rays=cali_bs,
            const_img_id=self.const_img_id,
            device=self.device,
            minpct=0.1,
            lossminpc=0.1,
        )

        self.grad_val = None
        self.loss_per_pix = None
        self.correction = None
        self.cur_points_2d = None
        self.alpha = args.soft_mining_alpha

        # cuda
        self.datasets.poses = self.datasets.poses.to(self.device)

    def sample_rays_and_rgbs(self):
        points_2d = self.lmc(self.grad_val, self.loss_per_pix)
        points_2d.round_()
        points_2d.requires_grad = True
        x, y = points_2d[:, 0], points_2d[:, 1]
        sample_rays, sample_rgbs = self.datasets.fetch_sample_rays(
            self.const_img_id, x, y
        )

        self.cur_points_2d = points_2d

        return sample_rays, sample_rgbs


    def compute_loss(self, rgb_map, rgb_train, iteration):
        ### follow softmining / train_ngp_nerf_prop.py
        loss_per_pix = ((rgb_map - rgb_train) ** 2).mean(-1)  # l2
        imp_loss = torch.abs(rgb_map - rgb_train).mean(-1).detach()  # l1
        correction = (
            1.0 / torch.clip(imp_loss, min=torch.finfo(torch.float16).eps).detach()
        )
        if self.alpha != 0:
            r = min((iteration / 1000), self.alpha)
        else:
            r = self.alpha
        correction.pow_(r)
        correction.clamp_(min=0.2, max=correction.mean() + correction.std())
        loss_per_pix.mul_(correction)

        self.loss_per_pix = loss_per_pix
        self.correction = correction

        loss = loss_per_pix.mean()
        return loss

    @torch.no_grad()
    def update_grad_val(self):
        net_grad = self.cur_points_2d.grad.detach()
        loss_per_pix = self.loss_per_pix.detach()
        net_grad = net_grad / (
            ((self.correction * loss_per_pix).unsqueeze(1))
            + torch.finfo(net_grad.dtype).eps
        )
        self.grad_val = net_grad




@torch.no_grad()
def export_mesh(args):
    device = args.device
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt["kwargs"]
    kwargs.update({"device": device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha, _ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(
        alpha.cpu(), f"{args.ckpt[:-3]}.ply", bbox=tensorf.aabb.cpu(), level=0.005
    )


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(
        args.datadir, split="test", downsample=args.downsample_train, is_stack=True
    )
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print("the ckpt path does not exists!!")
        return

    device = args.device
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt["kwargs"]
    kwargs.update({"device": device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = dataset(
            args.datadir, split="train", downsample=args.downsample_train, is_stack=True
        )
        PSNRs_test, avg_res = evaluation(
            train_dataset,
            tensorf,
            args,
            renderer,
            f"{logfolder}/imgs_train_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )
        print(
            f"======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================"
        )

    if args.render_test:
        os.makedirs(f"{logfolder}/{args.expname}/imgs_test_all", exist_ok=True)
        evaluation(
            test_dataset,
            tensorf,
            args,
            renderer,
            f"{logfolder}/{args.expname}/imgs_test_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f"{logfolder}/{args.expname}/imgs_path_all", exist_ok=True)
        evaluation_path(
            test_dataset,
            tensorf,
            c2ws,
            renderer,
            f"{logfolder}/{args.expname}/imgs_path_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )


def reconstruction(args, recorder):

    # init dataset
    device = args.device
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(
        args.datadir, split="train", downsample=args.downsample_train, is_stack=False
    )
    test_dataset = dataset(
        args.datadir, split="test", downsample=args.downsample_train, is_stack=True
    )
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    logfolder = os.path.join(args.basedir, args.expname, args.nerf_scene)

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_rgba", exist_ok=True)
    os.makedirs(f"{logfolder}/rgba", exist_ok=True)
    # summary_writer = SummaryWriter(logfolder)
    summary_writer = writer.inst

    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt["kwargs"]
        kwargs.update({"device": device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(
            aabb,
            reso_cur,
            device,
            density_n_comp=n_lamb_sigma,
            appearance_n_comp=n_lamb_sh,
            app_dim=args.data_dim_color,
            near_far=near_far,
            shadingMode=args.shadingMode,
            alphaMask_thres=args.alpha_mask_thre,
            density_shift=args.density_shift,
            distance_scale=args.distance_scale,
            pos_pe=args.pos_pe,
            view_pe=args.view_pe,
            fea_pe=args.fea_pe,
            featureC=args.featureC,
            step_ratio=args.step_ratio,
            fea2denseAct=args.fea2denseAct,
        )

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # linear in logrithmic space
    N_voxel_list = (
        torch.round(
            torch.exp(
                torch.linspace(
                    np.log(args.N_voxel_init),
                    np.log(args.N_voxel_final),
                    len(upsamp_list) + 1,
                )
            )
        ).long()
    ).tolist()[1:]

    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if (not args.ndc_ray) and args.filter_rays:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)

    if args.strategy == "full" or args.strategy == "random":
        trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)
    elif args.strategy == "expansive":
        trainingSampler = ExpansiveSupSampler(train_dataset, args)
    elif args.strategy == "soft":
        trainingSampler = SoftminingSampler(train_dataset, args)
    elif args.strategy == "egra":
        trainingSampler = EgraSampler(train_dataset, args)
    else:
        raise NotImplementedError

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    pbar = tqdm(
        range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout
    )
    for iteration in pbar:
        torch.cuda.synchronize()
        log.start_timer("train")
        log.start_timer("sampler")

        _reset_randomness()
        if args.strategy == "soft":
            rays_train, rgb_train = trainingSampler.sample_rays_and_rgbs()
        else:
            ray_idx = trainingSampler.nextids()
            rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx]
        _reset_fixed_seed(args.seed)

        torch.cuda.synchronize()
        log.pause_timer("train")
        log.pause_timer("sampler")

        rays_train, rgb_train = rays_train.to(device), rgb_train.to(device)

        log.start_timer("train")
        # rgb_map, alphas_map, depth_map, weights, uncertainty
        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(
            rays_train,
            tensorf,
            chunk=args.batch_size,
            N_samples=nSamples,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
            is_train=True,
        )

        if args.strategy == "soft" or args.strategy == "expansive":
            loss = trainingSampler.compute_loss(rgb_map, rgb_train, iteration)
        else:
            loss = torch.mean((rgb_map - rgb_train) ** 2)

        torch.cuda.synchronize()
        log.pause_timer("train")

        # loss
        total_loss = loss
        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight * loss_reg
            summary_writer.add_scalar(
                "train/reg", loss_reg.detach().item(), global_step=iteration
            )
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight * loss_reg_L1
            summary_writer.add_scalar(
                "train/reg_l1", loss_reg_L1.detach().item(), global_step=iteration
            )

        if TV_weight_density > 0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar(
                "train/reg_tv_density", loss_tv.detach().item(), global_step=iteration
            )
        if TV_weight_app > 0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg) * TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar(
                "train/reg_tv_app", loss_tv.detach().item(), global_step=iteration
            )

        log.start_timer("train")
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()

        if args.strategy == "soft":
            trainingSampler.update_grad_val()

        torch.cuda.synchronize()
        log.pause_timer("train")

        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar("train/PSNR", PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar("train/mse", loss, global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f"Iteration {iteration:05d}:"
                + f" train_psnr = {float(np.mean(PSNRs)):.2f}"
                + f" test_psnr = {float(np.mean(PSNRs_test)):.2f}"
                + f" mse = {loss:.6f}"
            )
            PSNRs = []

        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis != 0:
            PSNRs_test, eval_avg_res = evaluation(
                test_dataset,
                tensorf,
                args,
                renderer,
                f"{logfolder}/imgs_vis/",
                N_vis=args.N_vis,
                prtx=f"{iteration:06d}_",
                N_samples=nSamples,
                white_bg=white_bg,
                ndc_ray=ndc_ray,
                compute_extra_metrics=True,
            )

            summary_writer.add_scalar(
                "test/psnr", np.mean(PSNRs_test), global_step=iteration
            )

            # log
            log.inst.info(f"iters: {iteration} → psnr {eval_avg_res[0]:.6f}")
            log.inst.info(f"iters: {iteration} → ssim {eval_avg_res[1]:.8f}")
            log.inst.info(f"iters: {iteration} → lpips_a {eval_avg_res[2]:.8f}")
            log.inst.info(f"iters: {iteration} → lpips_v {eval_avg_res[3]:.8f}")

            # recorder
            recorder[iteration]["psnr"] = eval_avg_res[0]
            recorder[iteration]["ssim"] = eval_avg_res[1]
            recorder[iteration]["lpips_a"] = eval_avg_res[2]
            recorder[iteration]["lpips_v"] = eval_avg_res[3]

        log.start_timer("train")
        if iteration in update_AlphaMask_list:

            if (
                reso_cur[0] * reso_cur[1] * reso_cur[2] < 256**3
            ):  # update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)

            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                if args.filter_rays:
                    # filter rays outside the bbox
                    allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs)
                    trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)

        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(
                args.lr_init * lr_scale, args.lr_basis * lr_scale
            )
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        torch.cuda.synchronize()
        log.pause_timer("train")

    tensorf.save(os.path.join(logfolder, f"ckpt_{iteration}.th"))

    if args.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = dataset(
            args.datadir, split="train", downsample=args.downsample_train, is_stack=True
        )
        PSNRs_test, eval_avg_res = evaluation(
            train_dataset,
            tensorf,
            args,
            renderer,
            f"{logfolder}/imgs_train_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )
        print(
            f"======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================"
        )

    if args.render_test:
        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        PSNRs_test, eval_avg_res = evaluation(
            test_dataset,
            tensorf,
            args,
            renderer,
            f"{logfolder}/imgs_test_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )
        summary_writer.add_scalar(
            "test/psnr_all", np.mean(PSNRs_test), global_step=iteration
        )
        print(
            f"======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================"
        )

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print("========>", c2ws.shape)
        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
        evaluation_path(
            test_dataset,
            tensorf,
            c2ws,
            renderer,
            f"{logfolder}/imgs_path_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )


def train_tensorf(exter_args, recorder):
    args = merge_args(exter_args)

    _args = {k: v for k, v in vars(args).items() if k not in vars(exter_args)}
    with open(os.path.join(args.save_folder, "tensorf_args.json"), "w") as f:
        json.dump(_args, f, indent=4)

    if args.export_mesh:
        export_mesh(args)
    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args, recorder)


# if __name__ == '__main__':

#     torch.set_default_dtype(torch.float32)
#     torch.manual_seed(20211202)
#     np.random.seed(20211202)

#     args = config_parser()
#     print(args)

#     if args.export_mesh:
#         export_mesh(args)

#     if args.render_only and (args.render_test or args.render_path):
#         render_test(args)
#     else:
#         reconstruction(args)
