import configargparse
import copy
import os
from types import SimpleNamespace


TensoRFNameDic = {
    "syn": "blender",
    "nsvf": "nsvf",
    "llff": "llff",
    "tt": "tankstemple",
}

def merge_args(exter_args):
    args = copy.deepcopy(exter_args)  # Namespace
    
    #### default: syn settings

    args.expname = os.path.basename(args.save_folder)
    args.basedir = os.path.dirname(args.save_folder)

    args.add_timestamp = 0
    args.datadir = args.input_path
    args.progress_refresh_rate = 10
    args.with_depth = True
    args.downsample_train = 1.0
    args.downsample_test = 1.0
    args.model_name = "TensorVMSplit"
    args.batch_size = exter_args.batch_size
    args.n_iters = 20000 #30000
    args.dataset_name = TensoRFNameDic.get(args.nerf_dataset, "own_data")

    # learning rate
    args.lr_init = 0.02
    args.lr_basis = 1e-3
    args.lr_decay_iters = -1
    args.lr_decay_target_ratio = 0.1
    args.lr_upsample_reset = 1

    # loss
    args.L1_weight_inital = 8e-5
    args.L1_weight_rest = 4e-5
    args.Ortho_weight = 0.0
    args.TV_weight_density = 0.0
    args.TV_weight_app = 0.0

    # volume options
    args.n_lamb_sigma = [16, 16, 16]
    args.n_lamb_sh = [48, 48, 48]
    args.data_dim_color = 27
    args.rm_weight_mask_thre = 1e-4
    args.alpha_mask_thre = 0.0001
    args.distance_scale = 25
    args.density_shift = -10

    # network decoder
    args.shadingMode = "MLP_Fea" # "MLP_PE"
    args.pos_pe = 6
    args.view_pe = 2
    args.fea_pe = 2
    args.featureC = 128

    # output options
    args.ckpt = None
    args.render_only = 0
    args.render_test = 0 
    args.render_train = 0
    args.render_path = 0
    args.export_mesh = 0

    # rendering options
    args.lindisp = False
    args.perturb = 1.
    args.accumulate_decay = 0.998
    args.fea2denseAct = "softplus"
    args.ndc_ray = 0
    args.nSamples = 1e6
    args.step_ratio = 0.5

    ## blender(syn) flags
    args.white_bkgd = False
    args.N_voxel_init = 2097156 # 128**3
    args.N_voxel_final = 27000000 # 300**3
    args.upsamp_list = [2000, 3000, 4000, 5500, 7000]
    args.update_AlphaMask_list = [2000, 4000]
    args.idx_view = 0
    args.N_vis = 5
    args.vis_every = 5000  # 5000  #15000 # log metrics

    ## extra
    args.soft_mining_alpha = exter_args.soft_mining_alpha
    args.beta_a = exter_args.es_beta_a
    ### ablation
    args.use_mse = exter_args.use_mse
    args.a_s_add = exter_args.a_s_add
    args.lamda_a = exter_args.lamda_a
    args.lamda_s = exter_args.lamda_s

    ### modification
    args.filter_rays = False # stable sampling 



    #### specific settings
    if args.nerf_dataset == "nsvf":
        pass
    elif args.nerf_dataset == "llff":
        args.downsample_train = 4.0
        args.ndc_ray = 1
        args.N_voxel_init = 209715
        args.N_voxel_final = 262144000
        args.upsamp_list = [2000,3000,4000,5500]
        args.update_AlphaMask_list = [2500]
        args.n_lamb_sigma = [16,4,4]
        args.n_lamb_sh = [48,12,12]
        args.fea2denseAct = "relu"
        args.view_pe = 0
        args.fea_pe = 0
        args.TV_weight_density = 1.0
        args.TV_weight_app = 1.0

    return args


def config_parser(exter_args):
    ###### mannuly set exter_args

    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True, help="config file path")

    parser.add_argument("--expname", type=str, help="experiment name")

    parser.add_argument(
        "--basedir", type=str, default="./log", help="where to store ckpts and logs"
    )

    parser.add_argument(
        "--add_timestamp", type=int, default=0, help="add timestamp to dir"
    )
    parser.add_argument(
        "--datadir", type=str, default="./data/llff/fern", help="input data directory"
    )
    parser.add_argument(
        "--progress_refresh_rate",
        type=int,
        default=10,
        help="how many iterations to show psnrs or iters",
    )

    parser.add_argument("--with_depth", action="store_true")
    parser.add_argument("--downsample_train", type=float, default=1.0)
    parser.add_argument("--downsample_test", type=float, default=1.0)

    parser.add_argument(
        "--model_name",
        type=str,
        default="TensorVMSplit",
        choices=["TensorVMSplit", "TensorCP"],
    )

    # loader options
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_iters", type=int, default=30000)

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="blender",
        choices=["blender", "llff", "nsvf", "dtu", "tankstemple", "own_data"],
    )

    # training options
    # learning rate
    parser.add_argument("--lr_init", type=float, default=0.02, help="learning rate")
    parser.add_argument("--lr_basis", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--lr_decay_iters",
        type=int,
        default=-1,
        help="number of iterations the lr will decay to the target ratio; -1 will set it to n_iters",
    )
    parser.add_argument(
        "--lr_decay_target_ratio",
        type=float,
        default=0.1,
        help="the target decay ratio; after decay_iters inital lr decays to lr*ratio",
    )
    parser.add_argument(
        "--lr_upsample_reset",
        type=int,
        default=1,
        help="reset lr to inital after upsampling",
    )

    # loss
    parser.add_argument(
        "--L1_weight_inital", type=float, default=0.0, help="loss weight"
    )
    parser.add_argument("--L1_weight_rest", type=float, default=0, help="loss weight")
    parser.add_argument("--Ortho_weight", type=float, default=0.0, help="loss weight")
    parser.add_argument(
        "--TV_weight_density", type=float, default=0.0, help="loss weight"
    )
    parser.add_argument("--TV_weight_app", type=float, default=0.0, help="loss weight")

    # model
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, action="append")
    parser.add_argument("--n_lamb_sh", type=int, action="append")
    parser.add_argument("--data_dim_color", type=int, default=27)

    parser.add_argument(
        "--rm_weight_mask_thre",
        type=float,
        default=0.0001,
        help="mask points in ray marching",
    )
    parser.add_argument(
        "--alpha_mask_thre",
        type=float,
        default=0.0001,
        help="threshold for creating alpha mask volume",
    )
    parser.add_argument(
        "--distance_scale",
        type=float,
        default=25,
        help="scaling sampling distance for computation",
    )
    parser.add_argument(
        "--density_shift",
        type=float,
        default=-10,
        help="shift density in softplus; making density = 0  when feature == 0",
    )

    # network decoder
    parser.add_argument(
        "--shadingMode", type=str, default="MLP_PE", help="which shading mode to use"
    )
    parser.add_argument("--pos_pe", type=int, default=6, help="number of pe for pos")
    parser.add_argument("--view_pe", type=int, default=6, help="number of pe for view")
    parser.add_argument(
        "--fea_pe", type=int, default=6, help="number of pe for features"
    )
    parser.add_argument(
        "--featureC", type=int, default=128, help="hidden feature channel in MLP"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )
    parser.add_argument("--render_only", type=int, default=0)
    parser.add_argument("--render_test", type=int, default=0)
    parser.add_argument("--render_train", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=0)
    parser.add_argument("--export_mesh", type=int, default=0)

    # rendering options
    parser.add_argument(
        "--lindisp",
        default=False,
        action="store_true",
        help="use disparity depth sampling",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="set to 0. for no jitter, 1. for jitter",
    )
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument("--fea2denseAct", type=str, default="softplus")
    parser.add_argument("--ndc_ray", type=int, default=0)
    parser.add_argument(
        "--nSamples",
        type=int,
        default=1e6,
        help="sample point each ray, pass 1e6 if automatic adjust",
    )
    parser.add_argument("--step_ratio", type=float, default=0.5)

    ## blender flags
    parser.add_argument(
        "--white_bkgd",
        action="store_true",
        help="set to render synthetic data on a white bkgd (always use for dvoxels)",
    )

    parser.add_argument("--N_voxel_init", type=int, default=100**3)
    parser.add_argument("--N_voxel_final", type=int, default=300**3)
    parser.add_argument("--upsamp_list", type=int, action="append")
    parser.add_argument("--update_AlphaMask_list", type=int, action="append")

    parser.add_argument("--idx_view", type=int, default=0)
    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=5, help="N images to vis")
    parser.add_argument(
        "--vis_every", type=int, default=10000, help="frequency of visualize the image"
    )
    parsed_args = parser.parse_args()

    return parsed_args
