import argparse
import os
import json

from util.misc import gen_cur_time,gen_random_str


class Opt(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_base()
        self.add_inr()
        self.add_transform()
        self.add_sampler()

    def add_base(self):
        # base
        self.parser.add_argument("--seed", type=int, default=3407)
        self.parser.add_argument("--save_dir", type=str, default="./log")
        self.parser.add_argument("--tag", type=str, default="exp")
        self.parser.add_argument("--up_folder_name", type=str, default="debug")
        self.parser.add_argument("--debug", action="store_true")
        self.parser.add_argument("--device", type=str, default="cuda:0")

        # log
        self.parser.add_argument(
            "--log_epoch",
            type=int,
            default=500,
            help="log performance during training",
        )
        self.parser.add_argument(
            "--snap_epoch",
            type=int,
            default=5000,
            help="save fitted data during training",
        )

        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=4096,
            help="for radiance field which requires batch training",
        )

        # data folder
        self.parser.add_argument(
            "--data_folder",
            type=str,
            default="/home/ubuntu/projects/coding4paper/data")

    def add_inr(self):
        # trainer
        self.parser.add_argument(
            "--num_epochs", type=int, default=5000, help="Number of epochs"
        )

        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--use_lr_scheduler", action="store_true")

        self.parser.add_argument("--multi_data", action="store_true")

        # data
        self.parser.add_argument(
            "--input_path",
            type=str,
            help="can be dataset",
        )
        self.parser.add_argument(
            "--signal_type",
            type=str,
            default="image",
            choices=[
                "series",
                "audio",
                "image",
                "ultra_image",
                "shape",
                "radiance_field",
                "video",
            ],
        )
        # eval
        self.parser.add_argument("--eval_lpips", action="store_true")

        # model settings
        self.parser.add_argument(
            "--model_type",
            type=str,
            default="siren",
            choices=["siren", "finer", "wire", "gauss", "pemlp", "branch_siren"],
        )
        self.parser.add_argument(
            "--hidden_layers", type=int, default=3, help="hidden_layers"
        )
        self.parser.add_argument(
            "--hidden_features", type=int, default=256, help="hidden_features"
        )
        self.parser.add_argument(
            "--first_omega", type=float, default=30, help="(siren, wire, finer)"
        )
        self.parser.add_argument(
            "--hidden_omega", type=float, default=30, help="(siren, wire, finer)"
        )
        # Finer
        self.parser.add_argument(
            "--first_bias_scale",
            type=float,
            default=None,
            help="bias_scale of the first layer",
        )
        self.parser.add_argument("--scale_req_grad", action="store_true")
        # PEMLP
        self.parser.add_argument("--N_freqs", type=int, default=10, help="(PEMLP)")
        # Gauss, Wire
        self.parser.add_argument(
            "--scale", type=float, default=30, help="simga (wire, guass)"
        )

    def add_transform(self):
        # Please see https://weixiang-zhang.github.io/proj-symtrans/
        self.parser.add_argument(
            "--transform",
            type=str,
            default="min_max",
            choices=[
                "min_max",
                "z_score",
                "sym_power",
                "box_cox",
                "gamma",
            ],
        )
        self.parser.add_argument("--trans_shift", type=float, default=-0.5)
        self.parser.add_argument("--trans_scale", type=float, default=2.0, help="scale")

        self.parser.add_argument(
            "--pn_cum",
            type=float,
            default=0.5,
            help="cdf_index for power normalization",
        )
        self.parser.add_argument(
            "--pn_buffer",
            type=float,
            default=-1,
            help="buffer max and min val to solve long-tail | if == -1 → adpative | 0 == no work | >0 assign the value",
        )
        self.parser.add_argument(
            "--box_shift",
            type=float,
            default=0.1,
            help="+ shift*(max-min) to make it positive",
        )
        self.parser.add_argument(
            "--pn_alpha",
            type=float,
            default=0.01,
            help="hyper parmam of pn adaptive buffer",
        )
        self.parser.add_argument(
            "--pn_k",
            type=float,
            default=256.0,
            help="hyper parmam of pn adaptive buffer",
        )
        self.parser.add_argument(
            "--pn_beta",
            type=float,
            default=0.05,
            help="hyperparam for inverse edge calibiration",
        )

        self.parser.add_argument("--gamma_boundary", type=float, default=5)

        self.parser.add_argument(
            "--gamma", type=float, default=1.0, help="gamma for power normalization"
        )

        self.parser.add_argument(
            "--inverse", action="store_true", help="inverse pixels"
        )
        self.parser.add_argument(
            "--rpp", action="store_true", help="random pixel permutation"
        )


    def add_sampler(self):
        self.parser.add_argument(
            "--strategy",
            type=str,
            choices=["full", "random", "nmt", "soft", "expansive", "egra", "evos"],
            default="full"
        )

        self.parser.add_argument(
            "--sample_num_schedular",
            type=str,
            default="constant",
            help="lmc only can constant",
            choices=["constant", "linear", "step", "cosine", "reverse-cosine"],
        )

        # random & nmt & soft & our  → all strategies
        self.parser.add_argument("--use_ratio", type=float, default=0.5,help="it is also the init ratio of pixel ratio strategy")


        self.parser.add_argument("--profile_guide", type=str, default="value", choices=["value", "diff_1"])

        # soft mining
        self.parser.add_argument(
            "--soft_mining_alpha",
            type=float,
            default=0.6,
            help="see paper Accelerating Neural Field Training via Soft Mining",
        )

        self.parser.add_argument(
            "--soft_mining_a",
            type=float,
            default=2e1,
            help="see paper Accelerating Neural Field Training via Soft Mining",
        )

        self.parser.add_argument(
            "--soft_mining_b",
            type=float,
            default=2e-2,
            help="see paper Accelerating Neural Field Training via Soft Mining",
        )
        
        self.parser.add_argument(
            "--soft_mining_warmup",
            type=int,
            default=1000,
            help="see paper Accelerating Neural Field Training via Soft Mining",
        )
        self.parser.add_argument(
            "--wo_correction_loss",
            action="store_true"
        )
        
        self.parser.add_argument(
            "--soft_raw",
            action="store_true",
            help="Use points2D gradients",
        )

        # nmt
        
        self.parser.add_argument(
            "--nmt_profile_strategy",
            type=str,
            default="incremental",
            help="(strategy) how to profile NN",
            choices=[
                "incremental",
                "reverse-incremental",
                "dense",
                "dense2",
            ],
        )

        # corssover l1 & lap_l1
        self.parser.add_argument(
            "--crossover_method",
            type=str,
            default="select",
            choices=[
                "no",
                "add",
                "select"
            ],
        )
        
        # Cross-Frequency Supervision
        self.parser.add_argument(
            "--lap_coff",
            type=float,
            default=1e-5,
        )

        self.parser.add_argument(
            "--use_laplace_epoch",
            type=float,
            default=1500,
            help="save time",
        )

        # muatation
        self.parser.add_argument(
            "--mutation_method",
            type=str,
            default="constant",
            choices=[
                "constant",
                "linear",
                "exp"
            ],
        )

        self.parser.add_argument(
            "--init_mutation_ratio",
            type=float,
            default=0.5,
        )

        self.parser.add_argument(
            "--end_mutation_ratio",
            type=float,
            default=0.5,
        )
        
        # measurement
        self.parser.add_argument(
            "--measure_dense_diff", action="store_true", help="measure the difference of pixel, to guide profile frequency"
        )
        self.parser.add_argument(
            "--mdd_interval", type=int, default=1,
        )
        
        self.parser.add_argument(
            "--measure_crossover_diff", action="store_true", help="measure difference between l1 and lap"
        )
        ### only work for soft debugging and
        self.parser.add_argument(
            "--measure_sample_diff", action="store_true", help="measure sampling difference"
        )
        

        # evolution
        self.parser.add_argument(
            "--profile_interval_method",
            type=str,
            default="lin_dec",
            help="pim",
            choices=[
                "fixed",
                "lin_dec" 
            ],
        )

        self.parser.add_argument(
            "--init_interval",
            type=int,
            default=100,
        )
        self.parser.add_argument(
            "--end_interval",
            type=int,
            default=50,
        )

        # visualize
        self.parser.add_argument(
            "--record_indices", action="store_true"
        )
        self.parser.add_argument(
            "--dense_measure_psnr", action="store_true"
        )

    def _post_process(self, args):

        up_folder_name = args.up_folder_name
        if args.debug:
            up_folder_name = "debug"

        cur_time = gen_cur_time()
        random_str = gen_random_str()
        args.running_name = f"{args.tag}_{cur_time}_{random_str}"

        args.save_folder = os.path.join(
            args.save_dir, up_folder_name, args.running_name
        )
        os.makedirs(args.save_folder, exist_ok=True)

        # save to args.json
        with open(os.path.join(args.save_folder, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
            return args

    def get_args(self):
        args = self.parser.parse_args()
        args = self._post_process(args)
        return args
