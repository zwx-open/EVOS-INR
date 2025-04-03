# EVOS: Efficient Implicit Neural Training via EVOlutionary Selector

### [Project Page](https://weixiang-zhang.github.io/proj-evos/) | [Paper](https://arxiv.org/pdf/2412.10153) | [Code](https://github.com/zwx-open/EVOS-INR) | [Supplementary](https://weixiang-zhang.github.io/proj-evos/static/pdfs/03247_supp.pdf)

[Weixiang Zhang](https://weixiang-zhang.github.io/),
[Shuzhao Xie](https://shuzhaoxie.github.io/),
Chengwei Ren,
Siyi Xie,
[Chen Tang](https://www.chentang.cc/),
Shijia Ge,
Mingzi Wang,
[Zhi Wang*](http://pages.mmlab.top/)<br>
Tsinghua University \
\*: Corresponding Author

This is the official PyTorch implementation of "EVOS: Efficient Implicit Neural Training via EVOlutionary Selector" (accepted by *CVPR 2025*).

# Overview
<p align="center">
  <img src="./assets/pipeline6.png" style="width:100%;">
</p>

**Abstract.** We propose **EVOlutionary Selector (EVOS)**, an efficient training paradigm for accelerating Implicit Neural Representation (INR). Unlike conventional INR training that feeds all samples through the neural network in each iteration, our approach restricts training to strategically selected points, reducing computational overhead by eliminating redundant forward passes. Specifically, we treat each sample as an individual in an evolutionary process, where only those fittest ones survive and merit inclusion in training, adaptively evolving with the neural network dynamics. While this is conceptually similar to Evolutionary Algorithms, their distinct objectives (selection for acceleration vs. iterative solution optimization) require a fundamental redefinition of evolutionary mechanisms for our context. In response, we design sparse fitness evaluation, frequency-guided crossover, and augmented unbiased mutation to comprise EVOS. These components respectively guide sample selection with reduced computational cost, enhance performance through frequency-domain balance, and mitigate selection bias from cached evaluation. Extensive experiments demonstrate that our method achieves approximately 48%-66% reduction in training time while ensuring superior convergence without additional cost, establishing state-of-the-art acceleration among recent sampling-based strategies.


# Quick Start
## Clone Repository
```shell
git clone https://github.com/zwx-open/EVOS-INR
cd EVOS-INR
```
## Enviroment Setup
todo;

> **Tested Enviroments**: 
</br> - Ubuntu 20.04 with PyTorch 1.12.1 & CUDA 11.3.1 on RTX 3090

## Run Demo
> Demo: Fit `DIV2K/test/00.png` with SIREN + Sym-power-trans (5k epochs; ~5minutes)
```shell
python run.py 
```


# High Level Structure

# Code Execution
## How to run experiments in this repo? 
Please update code in `run.py` (~line 62) to run different experiments:
The defualt is running demo (exp_000):
```py
if __name__ == "__main__":

    exp = "000"
    param_sets = PAMRAM_SET[exp]
    gpu_list = [i for i in range(len(param_sets))]
    
    run_tasks(exp, param_sets, gpu_list)
```
For example, if you want to run experment 001, you can update it with `exp = "001"`. Moreover, feel free to allocate tasks for different gpu:
```py
if __name__ == "__main__":

    exp = "001"
    param_sets = PAMRAM_SET[exp]
    gpu_list = [3, 0] # assert len(param_sets) == len(gpu_list)
    
    run_tasks(exp, param_sets, gpu_list)
```
## How to set different tasks \& paramters?
For example, if you want to run *sym_power* and *01_norm* in `exp_001`, please update `config.py` with:

```python
PAMRAM_SET["001"] = (
            "01_norm",
            # "z_score",

            #"gamma_0.5",
            #"gamma_2.0",

            # "scale_0.5",
            # "scale_1.0",
            # "scale_2.0",

            # "inverse",
            # "rpp"
            # "box_cox",
            
            "sym_power",          
        )
```
## How to register new task?
Feel free to register new experiment by adding new key in `config.py`:
```python
PAMRAM_SET["999"] = (
            "xxx1",
            "xxx2",        
        )
```
and define corresponding function in `manager.py`:
```python
def _set_exp_999(self, param_set):
    if param_set == "xxx1":
        self.p.xxx = xxx1
    elif param_set == "xxx2":
        self.p.xxx = xxx2
```

## More Flexible Way
If you prefer a more flexible way to run this code, please refer to `debug.py`:
```py
'''flexiblely set all arugments in opt.py'''
def debug(use_cuda=0):
    args = [
        "--model_type",
        "siren",
        "--input_path",
        "./data/div2k/test_data/00.png",
        "--eval_lpips",
        "--transform",
        "sym_power",
        "--tag",
        "debug_demo",
        "--lr",
        "0.0001",
        "--up_folder_name",
        "000_demo", 
    ]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(use_cuda) 
    script = "python main.py " + " ".join(args)
    print(f"Running: {script}")
    os.system(script)
```

Then running `debug.py`:
```python
python debug.py
```

# Reproducing Results from the Paper

## Comparison of Different Transformations (Table 1)
<p align="center">
  <img src="./assets/table_1.png" style="width:90%;">
</p>

The setting of this experiments is correspoinding to `_set_exp_001()` in `manager.py`. Please run `exp_001` following [How to run experiments in this repo](#How-to-run-experiments-in-this-repo?).

Chekckpoints can be found in [here](https://drive.google.com/drive/folders/1VMtc84T4UsgoAluNKtOg-qoJXb1Z27q0?usp=drive_link) (`log/001_trans`).



# Citation
Please consider leaving a ⭐ and citing our paper if you find this project helpful:

```
@article{evos-inr,
  title={EVOS: Efficient Implicit Neural Training via EVOlutionary Selector},
  author={Zhang, Weixiang and Xie, Shuzhao and Ren, Chengwei and Xie, Siyi and Tang, Chen and Ge, Shijia and Wang, Mingzi and Wang, Zhi},
  journal={arXiv preprint arXiv:2412.10153},
  year={2024}
}
```

