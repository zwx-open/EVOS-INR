import os
from util.misc import gen_cur_time
import multiprocessing
from multiprocessing import Process, Queue
import subprocess


class TaskRunner(object):
    def __init__(self):
        self.cur_time = gen_cur_time()
        # os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

    def run_single_task(self, devices=(0,), nohup=False, just_get_cmd=False, **cmd_kw):
        assert len(devices) == 1
        device = devices[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        
        # for VSC debugger
        task_name = "debug" if just_get_cmd else cmd_kw.get("task_name","exp")
        if "task_name" in cmd_kw: del cmd_kw["task_name"]

        cmd = self.gen_task_cmd(
            tag=task_name,
            **cmd_kw,
        )
        
        if nohup: cmd = self._cmd2nohup(cmd)
        if not just_get_cmd: self._run_signle_cmd(cmd)

    def run_multiple_tasks(
        self,
        task_name="task",
        devices: tuple = (0, 1, 2, 3),
        dataset="syn",
        scene_list=[],
        **cmd_kw,
    ):
        raise NotImplementedError
        # save_folder_name = f"{dataset}_{self.cur_time}"
        # up_folder = os.path.join("./log", "tasks", task_name, save_folder_name)
        # cmd_queue = Queue()
        # datasetting = DatasetSetting[dataset]

        # if len(scene_list) == 0:
        #     scene_list = datasetting["scene_list"]

        # for scene in scene_list:
        #     cfg_path, data_path = TaskRunner._get_datasetting_path(dataset, scene)
        #     cmd = self.gen_task_cmd(
        #         tag=f"{scene}",
        #         save_dir=up_folder,
        #         my_config=cfg_path,
        #         datadir=data_path,
        #         **cmd_kw,
        #     )
        #     cmd_queue.put(cmd)

        # # EOF
        # for _ in devices:
        #     cmd_queue.put("")

        # all_procs = []
        # for gpu in devices:
        #     process = Process(target=self._process_func, args=(gpu, cmd_queue))
        #     process.daemon = True
        #     process.start()
        #     all_procs.append(process)
        # for i, gpu in enumerate(devices):
        #     all_procs[i].join()
        
        # print('up_folder: ', up_folder)
        # return up_folder

    def gen_task_cmd(self, **kw):
        
        # delete extra tasks
        # if "devices" in kw: del kw["devices"]

        args_dict = self._gen_default_args_dict(kw.get("tag", None))
        args_dict.update(**kw)
        cmd = self._convert_dict_to_cmd(args_dict)
        return cmd

    def _run_signle_cmd(self, cmd):
        print(f"Running cmd: {cmd}")

        ##### 1
        # process = subprocess.Popen(cmd, shell=True)
        # print(f"PID: {process.pid}")
        
        ##### 2
        os.system(cmd)

    def _gen_default_args_dict(self, tag="exp"):
        # warrper to opt.py
        default_args_dict = {
            "tag": tag,
            # "save_dir": f"./log/single/{tag}",
        }
        return default_args_dict
    
    def _process_func(self, device, queue):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        idx = 0
        while True:
            cmd = queue.get()
            if len(cmd) == 0:
                break
            cmd = self._cmd2back(cmd)
            # nohup 貌似会搞成并行
            # cmd = self._cmd2nohup(
            #     cmd, save_dir=os.path.join("out", self.cur_time), idx=idx
            # )
            print(f"Running {cmd} on {device}")
            proc_res = subprocess.run(
                cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            print(proc_res.stderr.decode())
            # print(proc_res.stdout.decode())

            # output = proc_res.stdout.decode(sys.stdout.encoding)

            # with open(f'out/{idx}_self.cur_time.out', 'w') as file_handler:
            #     file_handler.write(output)

            # subprocess.check_output(cmd, shell=True, env=env).decode(
            #     sys.stdout.encoding
            # )

            idx += 1


    def _convert_dict_to_cmd(self, args_dict):
        args_list = []

        for key, val in args_dict.items():
            args_list.append(f"--{key}")
            if val is not None:
                args_list.append(str(val))
        self._print_args_list(args_list)
        cmd = "python main.py " + " ".join(args_list)
        return cmd

    def _print_args_list(self, args_list):
        """for vscode debugger"""
        for item in args_list:
            print(f'"{item}",')

    def _cmd2nohup(self, cmd, save_dir="./out", idx=-1):
        preffix = f"{idx}_" if idx >= 0 else ""
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{preffix}{self.cur_time}.out")
        print(f"nohup | save on ./{save_path}")
        return f"nohup {cmd}  > {save_path} 2>&1 &"

    def _cmd2back(self, cmd):
        return f"{cmd} &"
