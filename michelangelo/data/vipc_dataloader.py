import torch
import numpy as np
import torch.utils.data as data

import os
import copy
import sys
sys.path.insert(0, os.path.dirname(__file__))
from dataset_utils import augment_cloud
from PIL import Image
from torchvision import transforms
import numpy as np
import pickle
import random
import math 
import open3d as o3d


class ViPCDataLoaderTest(data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.key = [0] * 100
    
    def __getitem__(self, index):
        # generate dummy data for testing purposes
        view = torch.zeros(3, 224, 224)
        pc = torch.zeros(3, 256)
        pc_part = torch.zeros(3, 128)
        label = torch.zeros(1)
        return pc, pc_part, view, label
    
    def __len__(self):
        return len(self.key)

class ViPCDataLoader(data.Dataset):
    def __init__(self, data_path, status, pc_input_num=4096, R=1, scale=1, image_size=224, 
                 augmentation=False, return_augmentation_params=False, debug=False, 
                 view_align=True, category='plane', mini=True):
        super(ViPCDataLoader,self).__init__()
        self.pc_input_num = pc_input_num
        self.status = status
        self.filelist = []
        self.cat = []
        self.key = []
        self.category = category
        self.view_align = view_align
        self.cat_map = {
            'plane':'02691156',
            'bench': '02828884', 
            'cabinet':'02933112', 
            'car':'02958343',
            'chair':'03001627',
            'monitor': '03211117',
            'lamp':'03636649',
            'speaker': '03691459', 
            'firearm': '04090263', 
            'couch':'04256520',
            'table':'04379243',
            'cellphone': '04401088', 
            'watercraft':'04530566'
        }
        filename = f"{status}_list.txt"
        with open(os.path.join(data_path, filename),'r') as f:
            line = f.readline()
            while (line):
                self.filelist.append(line)
                line = f.readline()
        
        self.imcomplete_path = os.path.join(data_path,'ShapeNetViPC-Partial')
        self.gt_path = os.path.join(data_path,'ShapeNetViPC-GT')
        self.rendering_path = os.path.join(data_path,'ShapeNetViPC-View')

        for key in self.filelist:
            if category !='all':
                if key.split('/')[0]!= self.cat_map[category]:
                    continue
            self.cat.append(key.split(';')[0])
            self.key.append(key)

        # self.key = np.load(os.path.join(data_path, f"ViPC_{status}_{category}_R{R}_all_names.npy"))
        # self.images = np.load(os.path.join(data_path, f"ViPC_{status}_{category}_R{R}_all_images.npy"))
        # self.pcs = np.load(os.path.join(data_path, f"ViPC_{status}_{category}_R{R}_all_completes.npy"))
        # self.pc_parts = np.load(os.path.join(data_path, f"ViPC_{status}_{category}_R{R}_all_partials.npy"))
        # self.key = [k.strip() for k in self.key]

        if debug:
            self.key = self.key[:5]
        elif mini:
            nsamples = 5000
            if status == "test":
                nsamples = int(nsamples * 0.3)
            self.key = random.sample(self.key, nsamples)

        # self.transform = transforms.Compose([
        #     transforms.Resize(image_size),
        #     # transforms.ToTensor(),
        #     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])

        self.train = status == "train"
        self.augmentation = augmentation  # augmentation could be a dict or False
        self.return_augmentation_params = return_augmentation_params

        # ---- label ----
        self.labels = np.full(shape=(len(self.key),), fill_value=R-1, dtype=np.int64)
        # ---- label ----

        self.scale = scale
        # self.input_data = self.input_data * scale
        # self.gt_data = self.gt_data * scale

        print('partial point clouds:', len(self.key))
        # if not benchmark:
        print('gt complete point clouds:', len(self.key))
        print('labels', len(self.labels))
        self.labels = self.labels.astype(int)
        self.R = 1
        self.npoints = self.pc_input_num
        self.image_size = image_size

    def fps_open3d(self, points: torch.Tensor):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        points = torch.from_numpy(np.array(pcd.points))
        normals = torch.from_numpy(np.array(pcd.normals))
        
        return points, normals

    def rotation_z(self, pts, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta, 0.0],
                                    [sin_theta, cos_theta, 0.0],
                                    [0.0, 0.0, 1.0]])
        return pts @ rotation_matrix.T


    def rotation_y(self, pts, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[cos_theta, 0.0, -sin_theta],
                                    [0.0, 1.0, 0.0],
                                    [sin_theta, 0.0, cos_theta]])
        return pts @ rotation_matrix.T


    def rotation_x(self, pts, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                    [0.0, cos_theta, -sin_theta],
                                    [0.0, sin_theta, cos_theta]])
        return pts @ rotation_matrix.T

    # def prepare_image(self, image_array):
    
    #     image_pt = torch.tensor(image).float()
    #     image_pt = rearrange(image_pt, "h w c -> c h w")
    
    #     return image_pt

    def __getitem__(self, idx):

        
        key = self.key[idx]
       
        pc_part_path = os.path.join(self.imcomplete_path,key.split('/')[0]+'/'+ key.split('/')[1]+'/'+key.split('/')[-1].replace('\n', '')+'.dat')
        # view_align = True, means the view of image equal to the view of partial points
        # view_align = False, means the view of image is different from the view of partial points
        if self.view_align:
            ran_key = key        
        else:
            ran_key = key[:-3]+str(random.randint(0,23)).rjust(2,'0')
       
        pc_path = os.path.join(self.gt_path, ran_key.split('/')[0]+'/'+ ran_key.split('/')[1]+'/'+ran_key.split('/')[-1].replace('\n', '')+'.dat')
        view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+'/'+ran_key.split('/')[1]+'/rendering/'+ran_key.split('/')[-1].replace('\n','')+'.png')
        
        #Inserted to correct a bug in the splitting for some lines 
        if(len(ran_key.split('/')[-1])>3):
            print("bug")
            print(ran_key.split('/')[-1])
            fin = ran_key.split('/')[-1][-2:]
            interm = ran_key.split('/')[-1][:-2]
            
            pc_path = os.path.join(self.gt_path, ran_key.split('/')[0]+'/'+ interm +'/'+ fin.replace('\n', '')+'.dat')
            view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+ '/' + interm + '/rendering/' + fin.replace('\n','')+'.png')

        views = Image.open(view_path).resize((self.image_size, self.image_size))
        views = np.array(views)
        views = views / 255 * 2 - 1
        views = views.transpose(2,1,0)
        views = views[:3,:,:]

        # load partial points
        with open(pc_path,'rb') as f:
            pc = pickle.load(f).astype(np.float32)
        # load gt
        with open(pc_part_path,'rb') as f:
            pc_part = pickle.load(f).astype(np.float32)
        # incase some item point number less than 3500 
        if pc_part.shape[0]<self.npoints:
            pc_part = np.repeat(pc_part,(self.npoints//pc_part.shape[0])+1,axis=0)[0:self.npoints]
        # assert pc_part.shape[0] == pc.shape[0]
        # assert pc_part.shape[0] == 3500
        # load the view metadata
        image_view_id = view_path.split('.')[0].split('/')[-1]
        part_view_id = pc_part_path.split('.')[0].split('/')[-1]
        
        view_metadata = np.loadtxt(view_path[:-6]+'rendering_metadata.txt')

        theta_part = math.radians(view_metadata[int(part_view_id),0])
        phi_part = math.radians(view_metadata[int(part_view_id),1])

        theta_img = math.radians(view_metadata[int(image_view_id),0])
        phi_img = math.radians(view_metadata[int(image_view_id),1])

        pc_part = self.rotation_y(self.rotation_x(pc_part, - phi_part),np.pi + theta_part)
        pc_part = self.rotation_x(self.rotation_y(pc_part, np.pi - theta_img), phi_img)

        # select npoints
        rng = np.random.default_rng()
        pc = pc[rng.choice(pc.shape[0], self.npoints, replace=True)]
        pc_part = pc_part[rng.choice(pc_part.shape[0], self.npoints, replace=True)]

        gt_mean = pc.mean(axis=0) 
        pc = pc -gt_mean
        pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
        pc = pc/pc_L_max

        pc_part = pc_part-gt_mean
        pc_part = pc_part/pc_L_max
        
        result = {}
        result['incomplete_points'] = (pc_part * self.scale).astype(np.float32)
        result['surface'] = (pc * self.scale).astype(np.float32)
        
        # augment the point clouds
        if (isinstance(self.augmentation, dict) and self.train):
            result_list = list(result.values())
            if self.return_augmentation_params:
                result_list, augmentation_params = augment_cloud(
                    result_list,
                    self.augmentation,
                    return_augmentation_params=True
                )
            else:
                result_list = augment_cloud(
                    result_list,
                    self.augmentation,
                    return_augmentation_params=False
                )
            for idx, key in enumerate(result.keys()):
                result[key] = result_list[idx]

        if self.return_augmentation_params:
            for key in augmentation_params.keys():
                result[key] = augmentation_params[key]
        for key in result.keys():
            result[key] = torch.from_numpy(result[key])
        
        # if(not self.train):
        result['name'] = copy.deepcopy(self.key[idx]).replace("/", "_")

        result['label'] = np.array(self.labels[idx])
        result['image'] = views
        result['text'] = torch.empty(1, 77)

        pc, pc_normals = self.fps_open3d(result['surface'])
        pc_part, pc_part_normals = self.fps_open3d(result['incomplete_points'])
        result['surface'] = torch.cat([pc, pc_normals], dim=-1)
        result['incomplete_points'] = torch.cat([pc_part, pc_part_normals], dim=-1)

        return result

    def __len__(self):
        return len(self.key)

if __name__ == "__main__":
    
    # import to pil_image function from torch transforms
    from torchvision.transforms.functional import to_pil_image
    data_dir = os.path.expanduser("~/Documents/datasets/ShapeNetViPC-Dataset")
    augmentation = {
            "pc_augm_scale": 1.2,
            "pc_augm_rot": True,
            "pc_rot_scale": 90,
            "pc_augm_mirror_prob": 0.5,
            "pc_augm_jitter": False,
            "translation_magnitude": 0.1,
            "noise_magnitude_for_generated_samples": 0
        }
    dataset = ViPCDataLoader(
        data_path=data_dir,
        status="train",
        augmentation=False,
    )
    
    outpath = "out_pointclouds/epoch_0029/original_dataloader_00.npz"
    data = [dataset[10]]
    for batch in data:
        surface = batch['incomplete_points']
        points = surface[:, :3]
        normals = surface[:, 3:]
        # save points
        np.savez(outpath, points=points, normals=normals)
        # save original as .xyz excluding normals 
        np.savetxt(outpath.replace(".npz", ".xyz"), batch['surface'][:, :3])
        np.savetxt(outpath.replace(".npz", "_sparse.xyz"), points)
        break
        # image = batch["image"]
        # print(image.shape, image.min(), image.max())
        # # normalize back
        # image = image.transpose(2, 1, 0)
        # image = (image * 0.5 + 0.5) * 255
        # image = image.astype(np.uint8)
        # print(image.shape, image.min(), image.max())
        # pil_image = Image.fromarray(image)
        # pil_image.save("out_pointclouds/epoch_0029/original_dataloader_00.jpg")
        # np.savez(outpath, points=points, normals=normals)
        # break
