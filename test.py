import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from models_prune import Generator_Prune

from datasets import ImageDataset
import datetime

#copy .pth to test environment
import moxing as mox
mox.file.copy("s3://models/horse2zebra/netG_B2A_prune_200.pth",'/tmp/log/horse2zebra/netG_B2A_prune_200.pth')
mox.file.copy_parallel("s3://models/GA/txt","/tmp/log/GA/txt/")



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/tmp/data/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', type=bool, default=True , help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

opt = parser.parse_args()
print(opt)

#construct mask
first_conv_out=64
mask_chns=[]
mask_chns.append(first_conv_out) #1st conv
mask_chns.append(first_conv_out*2) #2nd conv
mask_chns.append(first_conv_out*4) #3rd conv 1~9 res_block
mask_chns.append(first_conv_out*2) #1st trans_conv
mask_chns.append(first_conv_out) #2nd trans_conv
bit_len=0
for mask_chn in mask_chns:
    bit_len+= mask_chn





###### Definition of variables ######
# Networks
#pruning mask of generator
mask_input_B2A=np.loadtxt("/tmp/log/GA/txt/best_fitness.txt")
cfg_mask_B2A=compute_layer_mask(mask_input_B2A,mask_chns)

model_B2A = Generator_Prune(cfg_mask_B2A)
model_B2A.load_state_dict(torch.load('/tmp/log/horse2zebra/netG_B2A_prune_200.pth'))



if opt.cuda:
    model_B2A.cuda()
   

model_B2A.eval()



# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)

# Dataset loader

transform=transforms.Compose([
                              #transforms.Scale(originalSize),
                              #transforms.CenterCrop(imageSize),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) ])

###################################

###### Testing######
log_dir='/tmp/data/pruned_generators/'

root_dir='/tmp/data/images/'

# Create output dirs if they don't exist

list_all=os.listdir(root_dir)

for file in list_all:
    im=Image.open(root_dir+file).convert('RGB')
    im_tensor=transform(im)
    im_tensor=im_tensor.unsqueeze(0)
    im_tensor=im_tensor.cuda()
    fake_B = 0.5*(model_B2A(im_tensor).data + 1.0)
    new_name=file.replace('real_A','fake_B')
    save_image(fake_B, log_dir+new_name)
    
    


###################################
