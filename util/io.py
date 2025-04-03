# io and convert data type
# read | dump | cvt

import cv2 
import torch
import pickle
import json
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import imageio.v2 as imageio


def read_img_cv2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_img_pil(path):
    img = Image.open(path)
    #todo: 改为numpy
    return cvt_x_tensor(img) 

def read_img_io(path):
    img = imageio.imread(path)
    return np.array(img, dtype=np.float32)


def read_pkl(path, **kw):
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    return data

def read_json(path, **kw):
    with open(path, "r") as f:
        data = json.load(f)
    return data

#######################################################

def save_pkl(data, path, **kw):
    with open(path, "wb") as handler:
        pickle.dump(data, handler, protocol=pickle.HIGHEST_PROTOCOL) 
    return True

def save_json(data, path, **kw):
    with open(path, "wb") as handler:
        json.dump(data, handler, indent=4)

def save_cv2(img:np.ndarray, path, cvt_color=True):
    # 注意 opencv 存储 要求转换为unit8
    img = img.astype(np.uint8)
    # rgb to bgr (opencv默认bgr)
    if cvt_color: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    cv2.imwrite(path, img)

def save_img_io(img:np.ndarray, path):
    #转换为unit8
    img = img.astype(np.uint8)
    imageio.imwrite(path, img)

#######################################################
    
def cvt_cv2_pil(img):
    return Image.fromarray(img)
    
def cvt_x_tensor(data):
    return transforms.ToTensor()(data) # c,h,w
    
def cvt_np_tensor(np_arr):
    return torch.from_numpy(np_arr)

def cvt_tensor_np(torch_data):
    return torch_data.numpy()

def cvt_all_tensor(data):
    if torch.is_tensor(data):
        return data
    
    if isinstance(data, np.ndarray):
        return cvt_np_tensor(data)

    return cvt_x_tensor(data)

def cvt_cv2_torch(data:np.array):
    new_ = torch.tensor(data)
    if len(new_.shape) > 2:
        new_ = new_.permute(2,0,1) #  hwc →  chw 
    return new_

def cvt_torch_cv2(data:torch.tensor):
    if len(data.shape) > 2:
        new_ = data.permute(1,2,0) # chw →  hwc
    new_ =  new_.numpy()
    new_ = new_.astype(np.uint8)
    return new_




#######################################################
# def visualize_img(data:np.ndarray):
#     # 统一的转换方法 后面直接接 save_cv2 函数
#     assert len(data.shape) == 3 # h,w,c
#     # visualize_img = np.copy(data) 
#     if data.max() <= 1.0: # 模糊估计是0-1
#         data = data * 255.
#     return data