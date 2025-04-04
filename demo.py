import os

def demo(use_cuda=0):
    '''Args for EVOS(stepwise scheduler) with only one image'''
    args = [
        "--model_type",
        "siren",
        "--input_path",
        "./data/div2k/test_data/00.png",
        "--eval_lpips",
        "--log_epoch",
        "500",
        "--num_epochs",
        "5000",
        "--use_ratio",
        "0.2",
        "--strategy",
        "evos",
        "--sample_num_schedular",
        "step",
        "--tag",
        "evos",
        "--lr",
        "0.0001",
        "--up_folder_name",
        "000_demo_evos_stepwith",
    ]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(use_cuda) 
    script = "python main.py " + " ".join(args)
    print(f"Running: {script}")
    os.system(script)


if __name__ == "__main__":
    demo(0)
