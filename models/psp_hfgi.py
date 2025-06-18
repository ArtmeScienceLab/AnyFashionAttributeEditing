import matplotlib

matplotlib.use('Agg')
import torch
from torch import nn
from models.encoders import psp_encoders_hfgi as psp_encoders
# from models.stylegan2.model import Generator #, Discriminator
from models.stylegan2.networks import Generator, Discriminator
from configs.paths_config_twinnet import model_paths
import torchvision.transforms as transforms
import numpy as np
import os
import pickle
import cv2
import random

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt



def mask_direction(base_path):
    list_direction = os.listdir(base_path)
    seed = np.random.randint(low=0,high=len(list_direction),size=1)
    dic_name = seed + '.pkl'
    dic_path = os.path.join(base_path, dic_name)
    f = open(dic_path,'rb')
    direction = pickle.load(f)

    mask_name = seed + '.png' 
    mask_path = os.path.join(base_path, mask_name)
    mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE) 
    return mask, direction

class pSp(nn.Module):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.opts = opts
        # Define architecture
        # self.encoder = self.set_encoder()
        self.encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        self.residue =  psp_encoders.ResidualEncoder() #Ec
        # self.decoder = Generator(opts.stylegan_size, 512, 8, channel_multiplier=2)
        self.decoder = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=1024, square=False, img_channels=3)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # self.grid_transform = transforms.RandomPerspective(distortion_scale=opts.distortion_scale, p=opts.aug_rate)
        self.grid_align = psp_encoders.ResidualAligner() #ADA
        # self.discriminator = Discriminator(c_dim=0, img_resolution=1024, img_channels = 3)
        self.latent_avg = None
        # self.load_weights()

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print(self.opts.is_train)
            print('Loading basic encoder from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            torch.save(self.encoder.state_dict(), 'sketch_e4e_encoder.pt')

            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)

            if not self.opts.is_train:
                print('loading residue!!!!')
                self.residue.load_state_dict(get_keys(ckpt, 'residue'), strict=True)
                self.grid_align.load_state_dict(get_keys(ckpt, 'grid_align'), strict=True)
       
        else:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
        
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')

            ckpt = torch.load(self.opts.stylegan_weights)

            self.decoder.load_state_dict(ckpt, strict=False)
            self.__load_latent_avg(ckpt, repeat=self.encoder.style_count)

    def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            if self.opts.start_from_latent_avg:
                if codes.ndim == 2:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = not input_code

        dir_path = '/path_to/pretrained_models/fsmenv2_directions/image_00_region_overall_name_resefa.npy'
        wr = torch.from_numpy(np.load(dir_path)).to(self.opts.device).detach()
        wr = wr[0:codes.size()[0]]
        wr = wr.unsqueeze(1).repeat((1, 18, 1))

        is_manipulatable = torch.zeros(wr.shape, dtype=bool).to(self.opts.device)
        layer_index = [4,5,6,7]
        is_manipulatable[:, layer_index, :] = True
        linspace = np.array([6]*codes.size()[0])
        step_list = torch.tensor(linspace).to(self.opts.device, torch.float)
        step_list = step_list.to(torch.float32)
        step_list = step_list.unsqueeze(1)
        step_list = step_list.unsqueeze(1).repeat((1,18,512))
        inter_code = wr*step_list

        wr_add = torch.where(is_manipulatable, codes+inter_code, codes)
       
    
        edit, result_latent = self.decoder(wr_add, input_is_latent=True) 
        edit_resize = torch.nn.functional.interpolate(torch.clamp(edit, -1., 1.), size=(256,256) , mode='bilinear') 
     
        rec, result_latent = self.decoder(codes, input_is_latent=True) 
        rec_resize = torch.nn.functional.interpolate(torch.clamp(rec, -1., 1.), size=(256,256) , mode='bilinear') 
        
        codes1 = self.encoder(rec_resize)

        # codes_edit1 = self.encoder(rec_imgs_edit)

        if self.opts.start_from_latent_avg:
                if codes1.ndim == 2:
                    codes1 = codes1 + self.latent_avg.repeat(codes1.shape[0], 1, 1)[:, 0, :]
                    # codes_edit1 = codes_edit1 + self.latent_avg.repeat(codes_edit1.shape[0], 1, 1)[:, 0, :]
                else:
                    codes1 = codes1 + self.latent_avg.repeat(codes1.shape[0], 1, 1)
                    # codes_edit1 = codes_edit1 + self.latent_avg.repeat(codes_edit1.shape[0], 1, 1)
        
        wr_add1 = torch.where(is_manipulatable, codes1+inter_code, codes1)


        edit1, result_latent = self.decoder(wr_add1, input_is_latent=True)
        edit1_resize = torch.nn.functional.interpolate(torch.clamp(edit1, -1., 1.), size=(256,256) , mode='bilinear') 

        rec1, result_latent = self.decoder(codes1, input_is_latent=True)
        rec1_resize = torch.nn.functional.interpolate(torch.clamp(rec1, -1., 1.), size=(256,256) , mode='bilinear') 


        res_rec = x.detach() - rec_resize.detach() 
        res_edit = edit_resize.detach() - edit1_resize.detach() 
        res_edit_aligned = self.grid_align(torch.cat((res_rec, edit_resize), 1))
        res_rec_resize = torch.nn.functional.interpolate(torch.clamp(res_edit_aligned, -1., 1.), size=(256,128), mode='bilinear')
        res_edit_conditions = self.residue(res_rec_resize)

        if res_edit_conditions is not None:
            added_edit, result_latent = self.decoder(wr_add, res_edit_conditions, input_is_latent=True)
        
        added_edit_resize = torch.nn.functional.interpolate(torch.clamp(added_edit, -1., 1.), size=(256,256) , mode='bilinear') 
        res_edit =  added_edit_resize.detach() - edit1_resize.detach() 

        res_rec1_aligned = self.grid_align(torch.cat((res_edit, rec1_resize), 1))
        
        res_rec1_resize = torch.nn.functional.interpolate(torch.clamp(res_rec1_aligned, -1., 1.), size=(256,128), mode='bilinear')
        res_rec1_conditions = self.residue(res_rec1_resize)
        
        res_rec1 = x.detach() - rec1_resize.detach() 
        delta = res_rec1 - res_rec1_aligned
        if res_rec1_conditions is not None:
            added_rec, result_latent = self.decoder(codes1, res_rec1_conditions, input_is_latent=True)
        
        if resize:
            added_rec_resize = self.face_pool(added_rec)
        
        return added_rec_resize, delta

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

        print('latent_avg: ', self.latent_avg)
