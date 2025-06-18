import torch
import os
from models.psp_hfgi import pSp
import torchvision.transforms as transforms
from PIL import Image
import argparse
import warnings 
from tqdm import tqdm
import numpy as np
import time
warnings.filterwarnings("ignore")


def to_numpy(data):
    """Converts the input data to `numpy.ndarray`."""
    if isinstance(data, (int, float)):
        return np.array(data)
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    raise TypeError(f'Not supported data type `{type(data)}` for '
                    f'converting to `numpy.ndarray`!')

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

img_loadfold = transforms.Compose([
			transforms.Resize((256, 256)),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def tensor2im(var):
	# var shape: (3, H, W)
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))



def get_convert(net, base_dir, save_dir, direction, rate):
    img_list = os.listdir(base_dir)
    device = 'cuda'
    wr = torch.from_numpy(np.load(direction)).to(device)
    generator = net.decoder
    generator.eval()

    # times = []
    for i in tqdm(range(len(img_list))):
        img_path = os.path.join(base_dir, img_list[i])
        save_path = os.path.join(save_dir, img_list[i])
        np_name = img_list[i][:-4] + '.pt'
        save_w_path = os.path.join(save_dir, np_name)
       
        from_im = Image.open(img_path).convert('RGB')
        from_im = img_loadfold(from_im)
        from_im = from_im.unsqueeze(0).to(device)
        
        code = net.encoder(from_im)
        latent_code = code + net.latent_avg.repeat(code.shape[0], 1, 1)
 
        # print(save_w_path)
        # torch.save(latent_code.detach().cpu(), save_w_path)

        wr_add  = latent_code + rate * wr
        edit, _ = generator(wr_add ,input_is_latent=True)
        edit_resize = torch.nn.functional.interpolate(torch.clamp(edit, -1., 1.), size=(256,256) , mode='bilinear') 

        # torch.save(w.detach().cpu(), save_w_path)
        rec, _ = generator(latent_code ,input_is_latent=True)
        rec_resize = torch.nn.functional.interpolate(torch.clamp( rec, -1., 1.), size=(256,256) , mode='bilinear') 
        rec_x = (from_im - rec_resize).detach()
        res_edit_aligned = net.grid_align(torch.cat((rec_x, edit_resize), 1))
        rec_res = torch.nn.functional.interpolate(torch.clamp(res_edit_aligned, -1., 1.), size=(256,128) , mode='bilinear') 
        res_conditions = net.residue(rec_res)

        if res_conditions is not None:
            add_edit_images, result_latent = net.decoder(wr_add, res_conditions, input_is_latent=True)
            # print(add_edit_images.size())
            img = add_edit_images.squeeze()
            # exit()
        result = tensor2im(img)  
        result.save(save_path)
        # time_end = time.time()

   
    # print(np.mean(times))
        

def setup_model(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    # print('load ckpt: ',checkpoint_path)
    opts = ckpt['opts']
    net = pSp(opts)
    net.load_state_dict(ckpt['state_dict'], strict=True)
    if 'latent_avg' in ckpt:
        net.latent_avg = ckpt['latent_avg'].to(device)
    # print(net)
    net.eval()
    net = net.to(device)
    return net, opts


def main():

    device = "cuda"
    ckpt  = '/path_to/ckpt/fp/8_twinnet_h_product.pt' # path to the checkpoint
    net, opts = setup_model(ckpt, device)
    
    base_dir = '' # base directory
    save_dir = '' # save directory
    direction = '' # pair PCA direction
    rate = '' # editing rate, rate=0 means no editing only reconstruction

   
    get_convert(net, base_dir, save_dir, direction, rate)
    return


if __name__ == '__main__':
    main()