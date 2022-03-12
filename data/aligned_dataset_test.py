import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import linecache
import os.path as osp
import json
import numpy as np
import torch
from PIL import ImageDraw
class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.fine_height=256
        self.fine_width=192
        self.radius=5
        self.dataset_size = len(open('test_pairs.txt').readlines())

        dir_I = '_img'
        self.dir_I = os.path.join(opt.dataroot, opt.phase + dir_I)

        dir_C = '_color'
        self.dir_C = os.path.join(opt.dataroot, opt.phase + dir_C)

        dir_E = '_edge'
        self.dir_E = os.path.join(opt.dataroot, opt.phase + dir_E)

        dir_A = '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)

        dir_B = '_img'
        self.dir_B = os.path.join('/data/GDUT_student/chen_yang/data/base')

    def __getitem__(self, index):        

        file_path ='test_pairs.txt'
        im_name, c_name = linecache.getline(file_path, index+1).strip().split()

        I_path = os.path.join(self.dir_I,im_name)
        I = Image.open(I_path).convert('RGB')
        
        B_path = os.path.join(self.dir_B,im_name)
        B = Image.open(B_path).convert('RGB')

        params = get_params(self.opt, I.size)


        A_path = os.path.join(self.dir_A,im_name.split('.')[0]+'.png')
        A = Image.open(A_path).convert('L')
        transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        A_tensor = transform_A(A) * 255.0


        transform = get_transform(self.opt, params)
        transform_E = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

        I_tensor = transform(I)

        B_tensor = transform(B)

        C_path = os.path.join(self.dir_C,c_name)
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform(C)

        E_path = os.path.join(self.dir_E,c_name)
        E = Image.open(E_path).convert('L')
        E_tensor = transform_E(E)

        pose_name =I_path.replace('.png', '_keypoints.json').replace('.jpg','_keypoints.json').replace('test_img','test_pose')
        with open(osp.join(pose_name), 'r') as f:
            pose_label = json.load(f)
            try:
                pose_data = pose_label['people'][0]['pose_keypoints']
            except IndexError:
                pose_data = [0 for i in range(54)]
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = transform(one_map.convert('RGB'))
            pose_map[i] = one_map[0]
        P_tensor=pose_map
        input_dict = { 'image': I_tensor,'clothes': C_tensor, 'edge': E_tensor, 'label': A_tensor, 'pose':P_tensor, 'name':im_name, 'image_2':B_tensor}
        return input_dict

    def __len__(self):
        return self.dataset_size 

    def name(self):
        return 'AlignedDataset'
