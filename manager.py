from types import SimpleNamespace

STORE_TRUE = None
import os

DIV2K_TEST = "./data/div2k/test_data"
DIV2K_TRAIN = "./data/div2k/train_data"
KODAK = "./data/Kodak"
TEXT_TEST = "./data/text/test_data"
DEMO_IMG = os.path.join(DIV2K_TEST, "00.png")

DEMO_AUDIO = "../../data/libri_test_clean_121726/121-121726-0000.flac"
LIBRI_SLICE = "../../data/libri_test_clean_121726"

class ParamManager(object):
    def __init__(self, **kw):
        self._tag = "exp"
        self.p = SimpleNamespace()
        self._exp = ""

        self._set_exp(**kw)

    def _set_default_parmas(self):
        self.p.model_type = "siren"
        self.p.input_path = DEMO_IMG
        self.p.eval_lpips = STORE_TRUE
        self.p.warm_up = 0

    def _set_exp(self, idx=0, exp_num="000"):
        self._set_default_parmas()
        self.exp_num = exp_num
        eval(f"self._set_exp_{exp_num}(idx)")
        self.p.tag = f"{self._exp}"
        self.p.lr = self._get_lr_by_model(self.p.model_type)
        self.p.up_folder_name = self._tag

    def _convert_dict_args_list(self):
        args_dic = vars(self.p)
        args_list = []
        for key, val in args_dic.items():
            args_list.append(f"--{key}")
            if val is not STORE_TRUE:
                args_list.append(str(val))
        self._print_args_list(args_list)
        return args_list

    def export_args_list(self):
        return self._convert_dict_args_list()

    def export_cmd_str(self, use_cuda=[0]):
        args_list = self._convert_dict_args_list()
        script = "python main.py " + " ".join(args_list)
        script = self.add_cuda_visible_to_script(script, use_cuda)
        return script

    @staticmethod
    def add_cuda_visible_to_script(script, use_cuda=[0]):
        visible_devices: str = ",".join(map(str, use_cuda))
        return f"CUDA_VISIBLE_DEVICES={visible_devices} {script}"

    @staticmethod
    def _print_args_list(args_list):
        print("#" * 10 + "print for vscode debugger" + "#" * 10)
        for item in args_list:
            print(f'"{item}",')

    def _get_lr_by_model(self, model):
        if model == "gauss" or model == "wire":
            return 5e-3
        elif model == "siren":
            return 1e-4  # 1e-4 | 5e-4
        elif model == "finer":
            return 5e-4
        elif model == "pemlp":
            return 1e-3
        else:
            raise NotImplementedError

    def _use_single_data(self, pic_index="02", datasets = DIV2K_TEST):
        if hasattr(self.p, "multi_data"):
            delattr(self.p, "multi_data")

        self.p.input_path = os.path.join(datasets, f"{pic_index}.png")
        self._tag += f"_single_{pic_index}"

    def _use_datasets(self, type="div2k_test"):
        self.p.multi_data = STORE_TRUE
        if type == "div2k_test":
            self.p.input_path = DIV2K_TEST
        elif type == "div2k_train":
            self.p.input_path = DIV2K_TRAIN
        elif type == "kodak":
            self.p.input_path = KODAK
        elif type == "text_test":
            self.p.input_path = TEXT_TEST
        elif type == "libri_slice":
            self.p.input_path = LIBRI_SLICE
        self._tag += f"_{type}"


    def _set_fm_final2(self):
        self.p.strategy = "freeze"
        # profile
        self.p.profile_interval_method = "lin_dec"
        # self.p.init_interval = 75
        self.p.init_interval = 100
        self.p.end_interval = 50
        # crossover
        self.p.lap_coff = 1e-5
        self.p.crossover_method = "select"
        # mutation
        self.p.mutation_method = "constant"
        self.p.init_mutation_ratio = 0.5
        # self.p.init_mutation_ratio = 0.6
        # self.p.end_mutation_ratio = 0.4


    def _set_exp_001(self, _exp):
        use_ratio = 0.5
        self._tag = f"001_val_table1_constant_{use_ratio}"
        self._exp = f"{_exp}"

        self.p.log_epoch = 500 
        self.p.num_epochs = 5000
        self.p.use_ratio = use_ratio

        self._use_single_data("00")
        # self._use_datasets()

        self._by_sampler(_exp)
    
    def _set_exp_002(self, _exp):
        use_ratio = 0.2 # intial ratio
        self._set_exp_001(_exp)
        self.p.use_ratio = use_ratio
        self.p.sample_num_schedular = "step" 
        self._tag = f"002_table1_stepwise_{use_ratio}"
        
        self._use_single_data("00")
        # self._use_datasets()

    def _by_sampler(self, _exp,):
        if _exp == "full":
            self.p.strategy = "full"

        elif _exp == "random":
            self.p.strategy = "random"

        elif _exp == "fm_cur":
            self._set_fm_final()
        
        elif _exp == "fm_cur2":
            self._set_fm_final2()
        
        elif _exp == "fm_cur2_step":
            self._set_fm_final2()
            self.p.use_ratio = 0.2
            self.p.sample_num_schedular = "step"
        
        elif _exp == "fm_cur2_wo_crossover_laploss":
            self._set_fm_final2()
            self.p.lap_coff = 0

        elif _exp == "fm_cur_fix_mutation":
            self._set_fm_final()
            self.p.mutation_method = "constant"
            self.p.init_mutation_ratio = 0.6

        elif _exp == "fm_cur_fix_profile":
            self._set_fm_final()
            self.p.profile_interval_method = "fixed"
            self.p.init_interval = 75
        
        elif _exp.startswith("fm_cur2_fix_profile"):
            interval = _exp.split("_")[-1]
            self._set_fm_final2()
            self.p.profile_interval_method = "fixed"
            self.p.init_interval = interval

        elif _exp == "fm_cur_wo_crossover_all":
            self._set_fm_final()
            self.p.lap_coff = 0
            self.p.crossover_method = "no"

        elif _exp == "fm_cur_wo_crossover_laploss":
            self._set_fm_final()
            self.p.lap_coff = 0

        elif _exp == "fm_cur_wo_crossover_add":
            self._set_fm_final()
            self.p.crossover_method = "no"

        elif _exp == "fm_cur_wo_mutation":
            self._set_fm_final()
            self.p.mutation_method = "constant"
            self.p.init_mutation_ratio = 0
        
        elif _exp.startswith("fm_cur2_fix_mutation"):
            ratio = _exp.split("_")[-1]
            self._set_fm_final2()
            self.p.mutation_method = "constant"
            self.p.init_mutation_ratio = ratio

        elif _exp == "fm_cur_wo_profile":
            self._set_fm_final()
            self.p.profile_interval_method = "fixed"
            self.p.init_interval = self.p.num_epochs

        elif _exp == "fm_cur_dense_profile":
            self._set_fm_final()
            self.p.profile_interval_method = "fixed"
            self.p.init_interval = 2

        # nmt
        elif _exp == "nmt_incre":
            self.p.strategy = "nmt"
            self.p.nmt_profile_strategy = "incremental"
        elif _exp == "nmt_incre_step":
            self.p.strategy = "nmt"
            self.p.nmt_profile_strategy = "incremental"
            self.p.use_ratio = 0.2
            self.p.sample_num_schedular = "step"

        elif _exp == "nmt_dense2":
            self.p.strategy = "nmt"
            self.p.nmt_profile_strategy = "dense2"
        elif _exp == "nmt_dense":
            self.p.strategy = "nmt"
            self.p.nmt_profile_strategy = "dense"

        elif _exp == "nmt_dense_step":
            self.p.strategy = "nmt"
            self.p.nmt_profile_strategy = "dense"
            self.p.use_ratio = 0.2
            self.p.sample_num_schedular = "step"
        
        elif _exp == "nmt_rev_incre":
            self.p.strategy = "nmt"
            self.p.nmt_profile_strategy = "reverse-incremental"

        # soft mining
        elif _exp == "soft":
            self.p.strategy = "soft"
        
        elif _exp == "soft_hard":
            self.p.strategy = "soft"
            self.p.soft_mining_alpha = 0
        
        elif _exp == "soft_imp":
            self.p.strategy = "soft"
            self.p.soft_mining_alpha = 1

        elif _exp == "soft_mse":
            self.p.strategy = "soft"
            self.p.wo_correction_loss = STORE_TRUE
        
        elif _exp == "soft_raw":
            self.p.strategy = "soft"
            self.p.soft_raw =  STORE_TRUE

        # expansive supervision
        elif _exp == "expansive":
            self.p.strategy = "expansive"
        
        elif _exp == "egra":
            self.p.strategy = "egra"

     