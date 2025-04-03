import os
from manager import ParamManager
import subprocess
import numpy as np
import math

def run_subprocess(idx_list, gpu_list, exp_num):
    processes = []

    # assert len(idx_list) == len(gpu_list)
    _len = min(len(idx_list), len(gpu_list))
    idx_list = idx_list[:_len]
    gpu_list = gpu_list[:_len]

    for idx, use_cuda in zip(idx_list, gpu_list):
        pm = ParamManager(idx=idx, exp_num=exp_num)
        cmd_str = pm.export_cmd_str(use_cuda=[use_cuda])
        ##  print cmd str for debugger
        # exit()
        process = subprocess.Popen(cmd_str, shell=True)
        print(f"PID: {process.pid}")
        processes.append(process)

    for process in processes:
        process.wait()


def run_tasks(exp_num, param_idxs, gpu_list):

    gpus = len(gpu_list)
    rounds = math.ceil(len(param_idxs) / gpus)
    print("rounds: ", rounds)

    for i in range(rounds):
        cur_param_idxs = param_idxs[i * gpus : min(len(param_idxs), (i + 1) * gpus)]
        cur_len = len(param_idxs)
        gpu_list = gpu_list[:cur_len]
        run_subprocess(cur_param_idxs, gpu_list, exp_num)


if __name__ == "__main__":
   
    '''Replicate for Table_1 (Constant Scheduler)'''
    # param_idxs = [
    #     "full",
    #     "random",
    #     "egra",
    #     "expansive",
    #     "soft",
    #     "nmt_incre", 
    #     # "nmt_dense",
    #     "fm_cur2",
    #     "fm_cur2_wo_crossover_laploss",
    # ]
    # gpu_list = [0,1,2,3,4,5,6,7]
    # run_tasks("001", param_idxs, gpu_list)

    '''Replicate for Table_1 (Step-wise Scheduler)'''
    param_idxs = [
        "random",
        "egra",
        "expansive",
        "nmt_incre", 
        "fm_cur2",
        "fm_cur2_wo_crossover_laploss",
    ]
    gpu_list = [0,1,2,3,4,5,6,7]
    run_tasks("002", param_idxs, gpu_list)