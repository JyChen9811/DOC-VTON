import time
from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint,Refine
from models.afwm import AFWM
from models.unet2 import UNet,UNet1,UNet2
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F
import ipdb
from torch.autograd import Variable
from util import util
#from pylab import *
from sklearn.ensemble import IsolationForest
from PIL import Image
NC=14


def generate_discrete_label(inputs, label_nc, onehot=True, encode=True):
    pred_batch = []
    size = inputs.size()
    for input in inputs:
        input = input.view(1, label_nc, size[2], size[3])
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_map = []
    for p in pred_batch:
        p = p.view(1, 256, 192)
        label_map.append(p)
    label_map = torch.stack(label_map, 0)
    if not onehot:
        return label_map.float().cuda()
    size = label_map.size()
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)

    return input_label


def gen_noise(shape):
        noise = np.zeros(shape, dtype=np.uint8)
        ### noise
        noise = cv2.randn(noise, 0, 255)
        noise = np.asarray(noise / 255, dtype=np.uint8)
        noise = torch.tensor(noise, dtype=torch.float32)
        return noise.cuda()
def morpho(mask,iter,bigger=True):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    new=[]
    for i in range(len(mask)):
        tem=mask[i].cpu().detach().numpy().squeeze().reshape(256,192,1)*255
        tem=tem.astype(np.uint8)
        if bigger:
            tem=cv2.dilate(tem,kernel,iterations=iter)
        else:
            tem=cv2.erode(tem,kernel,iterations=iter)
        tem=tem.astype(np.float64)
        tem=tem.reshape(1,256,192)
        new.append(tem.astype(np.float64)/255.0)
    new=np.stack(new)
    new=torch.FloatTensor(new).cuda()
    return new
def encode_input(label_map, nc = NC):
    size = label_map.size()
    oneHot_size = (size[0], nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    #ipdb.set_trace()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
    #ipdb.set_trace()

    input_label = Variable(input_label)

    return input_label
def generate_label_plain(inputs, nc = NC):
    size = inputs.size()
    pred_batch = []
    for input in inputs:
        input = input.view(1, nc, 256,192)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_batch = pred_batch.view(size[0], 1, 256,192)

    return label_batch
def generate_label_color(inputs):
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], opt.label_nc))
    label_batch = np.array(label_batch)
    label_batch = label_batch * 2 - 1
    input_label = torch.from_numpy(label_batch)

    return input_label

def ger_average_color(mask,arms):
    color=torch.zeros(arms.shape).cuda()
    for i in range(arms.shape[0]):
        count = len(torch.nonzero(mask[i, :, :, :]))
        if count < 10:
            color[i, 0, :, :]=0
            color[i, 1, :, :]=0
            color[i, 2, :, :]=0

        else:
            color[i,0,:,:]=arms[i,0,:,:].sum()/count
            color[i,1,:,:]=arms[i,1,:,:].sum()/count
            color[i,2,:,:]=arms[i,2,:,:].sum()/count
    return color
opt = TestOptions().parse()

start_epoch, epoch_iter = 1, 0
creationL1 = nn.L1Loss()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print(dataset_size)


warp_model = AFWM(opt, 3)
print(warp_model)
warp_model.eval()
warp_model.cuda()
load_checkpoint(warp_model, '......./PFAFN_warp_epoch_101.pth')

gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
print(gen_model)
gen_model.eval()
gen_model.cuda()
load_checkpoint(gen_model,  '......./PFAFN_gen_epoch_040.pth')

comp_mask_model = UNet(in_channels=22, n_classes=3)
load_checkpoint(comp_mask_model , '......./comp_mask_epoch_081.pth')
comp_mask_model .eval()
comp_mask_model .cuda()

comp_occu_model = UNet(in_channels=32)
load_checkpoint(comp_occu_model , '......./comp_sleeve_epoch_final.pth')
comp_occu_model.eval()
comp_occu_model.cuda()






total_steps = (start_epoch-1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size / opt.batchSize

for epoch in range(1,2):
    l1_loss = 0
    for i, data in enumerate(dataset, start=epoch_iter):

        real_image = data['image']
        clothes = data['clothes']
        t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float))
        data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
        label = data['label'] * (1 - t_mask) + t_mask * 4
        pose = data['pose']
        ##edge is extracted from the clothes image with the built-in function in python
        edge = data['edge']
        edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
        clothes = clothes * edge        
        pose = data['pose']
        #image_test = data['image_test']
        keep_label = torch.FloatTensor((data['label'].cpu().numpy()==11).astype(np.int) + (data['label'].cpu().numpy()==13).astype(np.int)).cuda()

        person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy()==4).astype(np.int))
        label = label * (1-person_clothes_edge)
        shape = person_clothes_edge.size()
        face = torch.FloatTensor((data['label'].cpu().numpy() == 12).astype(np.int))
        flow_out = warp_model(real_image.cuda(), clothes.cuda())
        warped_cloth, last_flow, = flow_out
        warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                          mode='bilinear', padding_mode='zeros')
        person_arm_img = keep_label * real_image.cuda()
        warped_edge_un = torch.FloatTensor((warped_edge.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()

        warped_edge = torch.FloatTensor((warped_edge.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()

        #image_test_cloth = image_test.cuda() * warped_edge


        occu_skin_mask = person_clothes_edge.cuda() * (1 - warped_edge_un)
        arm_label = torch.FloatTensor((data['label'].cpu().numpy()==11).astype(np.float) *11+ (data['label'].cpu().numpy()==13).astype(np.float)*13).cuda()
        keep_skin_mask = arm_label * (1-warped_edge_un)
        label = label.cuda() * (1-warped_edge_un)
        keep_arm_label = (data['label'].cpu().numpy()==11).astype(np.float) + (data['label'].cpu().numpy()==13).astype(np.float)

        arm_image = keep_arm_label * real_image.detach().cpu().numpy()

        keep_arm_label = torch.FloatTensor(keep_arm_label).cuda()
      
        #occu_w_arm_label = keep_skin_mask + occu_skin_mask*4
        occu_w_arm_label = label + occu_skin_mask*4
        encode_occu_w_arm_label = encode_input(occu_w_arm_label)
        gen_arm_mask1 = comp_occu_model(torch.cat([pose.cuda(), encode_occu_w_arm_label],1))
        gen_arm_mask = torch.sigmoid(gen_arm_mask1)
        

        gen_arm_mask = generate_discrete_label(gen_arm_mask.detach(), 14, False)

        gen_arm_1 =  torch.FloatTensor(((gen_arm_mask.cpu().numpy()==11)).astype(np.float)).cuda()
        gen_arm_2 =  torch.FloatTensor(((gen_arm_mask.cpu().numpy()==13)).astype(np.float)).cuda() 
        deoccu_w_arm_label = gen_arm_mask * (1-gen_arm_1.cuda())
        deoccu_w_arm_label = deoccu_w_arm_label  * (1-gen_arm_2.cuda()) 
        clo_dilate = warped_edge_un * keep_arm_label
        clo_dilate = morpho(warped_edge_un,  8)
        clo_dilate = torch.FloatTensor((clo_dilate.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()
        gen_arm_1_dilate = gen_arm_1*(1-clo_dilate)
        gen_arm_2_dilate = gen_arm_2*(1-clo_dilate)
        gen_arm_1_dilate = morpho(gen_arm_1_dilate,30)
        gen_arm_2_dilate = morpho(gen_arm_2_dilate,30)
        gen_arm_1_dilate =  torch.FloatTensor(((gen_arm_1_dilate.detach().cpu().numpy() > 0.5)).astype(np.float)).cuda()
        gen_arm_2_dilate =  torch.FloatTensor(((gen_arm_2_dilate.detach().cpu().numpy() > 0.5)).astype(np.float)).cuda() #本来0.3

        gen_arm_1 = gen_arm_1*gen_arm_1_dilate
        gen_arm_2 = gen_arm_2*gen_arm_2_dilate
 
        gen_arm_1 = gen_arm_1* (1-warped_edge_un)  
        gen_arm_2 = gen_arm_2* (1-warped_edge_un)     
          
 
        deoccu_w_arm_label = gen_arm_1 * 1  +gen_arm_2 * 2 + warped_edge_un*3
        encode_deoccu_w_arm_label = encode_input(deoccu_w_arm_label, 4)
        deoccu_arm_mask = comp_mask_model (torch.cat([pose.cuda(), encode_deoccu_w_arm_label],1))
        deoccu_arm_mask = torch.sigmoid(deoccu_arm_mask)
        deoccu_arm_mask = generate_discrete_label(deoccu_arm_mask.detach(), 3, False)
        arm_1 =  torch.FloatTensor(((deoccu_arm_mask.cpu().numpy()==1)).astype(np.float)).cuda()
        arm_2 =  torch.FloatTensor(((deoccu_arm_mask.cpu().numpy()==2)).astype(np.float)).cuda()

        warped_edge_un[arm_1==1]=0
        warped_edge_un[arm_2==1]=0
        gen_arm_label = arm_1 + arm_2 - (arm_1 * arm_2)
        warped_cloth_un = warped_cloth * warped_edge_un

        img_deal = real_image.cuda() * (1 - person_clothes_edge.cuda()) * (1 - warped_edge_un)


        gen_inputs = torch.cat([real_image.cuda(), warped_cloth_un, warped_edge_un], 1)
        gen_outputs = gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge_un
        p_tryon = warped_cloth_un * m_composite + p_rendered * (1 - m_composite)
        




        path = 'results/' + opt.name
        os.makedirs(path, exist_ok=True)
        sub8_path = 'P_person_new'


        os.makedirs(sub8_path,exist_ok=True)
        #ipdb.set_trace()
        if step % 1 == 0:
            c = p_tryon.float().cuda()


            combine = c[0]
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.imwrite(sub8_path+'/'+str(step)+'.jpg',bgr)  






        step += 1
        if epoch_iter >= dataset_size:
            break



