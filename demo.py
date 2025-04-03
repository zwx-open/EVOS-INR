import os
# from manager import ParamManager
import subprocess
import numpy as np
import math



def demo(use_cuda=0):
   args =[
        "--up_folder_name", "000_demo",
        "--tag", "siren_00",
        "--num_epochs", 
        "5000",
        "--lr",
        "1e-4",  # gauss wire 5e-3 | siren finer 1e-4 | pemlp 1e-3
        "--model_type",
        "siren",
        "--input_path",
        "./data/div2k/test_data/00.png"

   ]
   os.environ['CUDA_VISIBLE_DEVICES'] = str(use_cuda) # '0,1'
   script = "python main.py " + ' '.join(args)
   print(f"Running: {script}")
   os.system(script)

if __name__ == '__main__':
   demo(0)