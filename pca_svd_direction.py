import configs.data_configs_twinnet as data_configs
from datasets.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from utils.model_utils import setup_model
import argparse
import sys
import torch
import os
import numpy as np
import imageio

sys.path.append(".")
sys.path.append("..")

# collar_test_num = ['416', '521', '1234', '1279', '2335', '2545', '2923', '3074', '3309', 
#             '3932', '4723', '4870', '5625', '5732', '6002', '6193', '6206', '6231',
#             '6974', '7179', '7595', '7679', '7843', '8159', '8284', '8579', '8849', 
#             '292745', '300206', '301684']

collar_test_num = ['240278','240615','240684','240720','240786','240989','241468','242352','242454',
                   '242588','243066','243134','243199','243336','243404','243573','243758','243792',
                   '243880','243914','243984','252515','252624','252754','252892','253094','253164',
                   '253298','253365','253400']

# bottom_test_num = ['100', '159', '418', '524', '852', '1866', '2324', '2794', '3309', '3687',
#              '3926', '4191', '4395', '5672', '5945', '6172', '7244', '7431', '7500', '7576',
#              '8248', '9259', '9352', '9702', '10751', '11012', '11358', '11430', '11655', '15673'
# ]

bottom_test_num=['240027','240031','240054','240065','240095','240099','240129','240201','240235','240269',
                 '240303','240333','240371','240439','240503','240507','240541','240639','240643','240707',
                 '240711','240745','240877','240901','240915','241017','241051','241207','241221','241285']

# sleeve_length_test_num=['521', '607', '780', '821', '847', '887', '1283', '1599', '1750', '2574', '2666', 
#                 '2821', '2923', '3074', '3076', '3133', '3149', '3228', '4272', '4332', '4424', '4711', 
#                 '4966', '5178', '5240', '5522', '7382', '7843', '8470', '8606']

sleeve_length_test_num=['231761','231925','233319','240271','240509','241478','241670','241873','253091','253259',
                        '276669','298813','299109','299289','299336','299445','299773','300619','300645','300809',
                        '301026','301193','301249','301476','301510','301730','302047','302566','302640','302956']

dress_length_test_num = ['3980','4117','4315','4579','4841','5073','5176','5299','6910','8648','8814','8968',
                         '10323','10650','10823','11126','11231','16985','19907','21140','24303','25389','41940',
                         '42435','49466','50628','51840','53192','54484','54963']

opentoclose_test_num = ['18954','22215','25816','26366','28418','29907','30128','31058','33497','37128','38888',
                          '43296','43555','43987','44455','45481','45890','54900','55563','58025','58753','60086',
                          '61819','64598','68052','71375','75985','76423','81570','89977']

# sleeve_shape_test_num = ['72','655','752','846','933','1733','1922','2488','4515','6206','8849','9671','10698',
#                          '11818','12039','14514','14659','16962','20351','23281','23336','24463','26346','32467',
#                          '32470','32526','33026','33329','37583','40259']
sleeve_shape_test_num = ['240271','240509','240567','240815','241091','241121','241207','241670','241873','242063',
                         '252790','252913','252971','253162','253259','253327','253413','253463','253519','253637',
                         '275903','276129','276159','276379','276481','276641','276669','286029','302597','302763',]

# top_length_test_num = ['727','3192','3243','4824','4870','5107','5394','7518','9237','9778','9779','10371','11655','11817',
#                        '14595','15390','17889','26444','26449','27017','28465','30441','32349','33047','35485','36342',
#                        '36403','37583','39290','41823']
# top_length_test_num=['240271','240509','240567','240815','241091','241121','241207','241670','241873','242063',
#                      '252790','252913','252971','253162','253259','253327','253413','253519','253637','275903',
#                      '276129','276159','276379','276481','276641','276669','286029','302597','302763'
# ]

top_length_test_num = ['1','2','3','4','5','6','7','8','9','10',
                       '11','12','13','14','15','16','17','18','19','20',
                       '21','22','23','24','25','26','27','28','29',
                       ]

bottom_length_test_num = [#'1','2','3','4','5','6','7','8','9','10',
                    #    '11','12','13','14','15','16','17','18','19','20',
                    #    '21','22','23','24','25','26','27','28','29','30',
                       '41','42','43','44','45','46','47','48','49','50',
                       '51','52','53','54','55','56','57','58','59','60',
                       '61','62','63','64','65','66','67','68','69','70'
                       ]

dress_length_test_num = ['1','2','3','4','5','6','7','8','9','10',
                    '11','12','13','14','15','16','17','18','19','20',
                        '21','22','23','24','25','26','27','28','29','30',
                    #    '41','42','43','44','45','46','47','48','49','50',
                    #    '51','52','53','54','55','56','57','58','59','60',
                    #    '61','62','63','64','65','66','67','68','69','70'
                    ]
 
# test_num = dress_length_test_num

def normalize(data):
    # data[data>0]=1
    # data[data<0] = 0
    # print(data)
    mu = np.mean(data, axis = 0)
    sigma = np.std(data, axis = 0)
    # s = np.sum(data)
    # print('the sum of the difference: ', s)
    return (data-mu)/sigma

def collar_pair_direction(f1_path,f2_path, direction_name, test_num_list):
    save_dir = ''
    diff_mix = []
    f1_name = f1_path.split('/')[-1]
    f2_name = f2_path.split('/')[-1]
    print(f1_name, f2_name)
    test_num = test_num_list
    print(test_num)

    for i in range(len(test_num)):
        # img1_name = 'test_' + test_num[i] + '_' + f1_name +'.npy'
        # img2_name = 'test_' + test_num[i] + '_' + f2_name +'.npy'
        img1_name = test_num[i] + '_' + f1_name +'.npy'
        img2_name = test_num[i] + '_' + f2_name +'.npy'

        img1_path = os.path.join(f1_path, img1_name)
        img2_path = os.path.join(f2_path, img2_name)
        print(img1_path)
        print(img2_path)

        if os.path.exists(img1_path) and os.path.exists(img2_path):
            img1 = np.load(img1_path)
            img2 = np.load(img2_path)
            sub_diff = img1 - img2
            # print(sub_diff.shape)
            sub_diff2 = np.reshape(sub_diff, (18*512))
            sub_diff2 = normalize(sub_diff2)
            diff_mix.append(sub_diff2)
            print(diff_mix)
    
    diff_mix2 = np.array(diff_mix)
    print(diff_mix2.shape)
    try:
        u, sigma, vt = np.linalg.svd(diff_mix2)
        direction = np.expand_dims(np.reshape(vt[0,:], [18,512]), 0)
        # print(sigma, vt.shape, direction.shape)
        direc_name = f1_name + '_vs_' + f2_name + '.npy'
        save_dir = '/path_to/directions/svd_directions/'+ direction_name + '/' + f1_name
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 

        save_path = os.path.join(save_dir, direc_name)
        np.save(save_path, direction)
        print('successsful save: ', save_path)
    except:
        return

def obtain_latents(args):
    # args.ckpt = '/path_to/experiment/8_twinnet_fswv1_100000.pt'
    args.ckpt = '/path_to/experiment/8_twinnet_fp_140000.pt'
    # args.ckpt = '/path_to/experiment/8_twinnet_fsmenv3_190000.pt'
    net, opts = setup_model(args.ckpt, device)
    generator = net.decoder
    generator.eval()
    dir_list = os.listdir(args.images_dir)
    ori_image_dir = args.images_dir
    for i in range(len(dir_list)):
        args.images_dir = os.path.join(args.images_dir, dir_list[i])
        category_name = args.images_dir.split('/')[-2]
        print( args.images_dir, category_name)
        args, data_loader = setup_data_loader(args, opts)
        # save_base = '/path_to/directions/latent_codes2/' + category_name +'/'+ dir_list[i]
        save_base = '/path_to/directions/latent_codes2/' + category_name +'/'+ dir_list[i]
        get_all_latents(net, data_loader, save_base, args.n_sample)
        args.images_dir = ori_image_dir


def test_direction2(latent_path, direction_path, mp4_path, generator, step_list):
    direction =  np.load(direction_path)
    direction = torch.from_numpy(direction).cuda()
    latent = np.load(latent_path)
    w_pivot = torch.from_numpy(latent).cuda()

    # direction = torch.load(direction_path, encoding = 'latin1').cuda()
    # w_pivot = torch.load(latent_path, encoding = 'latin1').cuda()
    frames = []
    # img, _ = generator(latent_code,input_is_latent=True)
    with torch.no_grad():
        for alpha in step_list:
            print('alpha', alpha)
            w = w_pivot + alpha * direction
            img, _ = generator(w,input_is_latent=True)
            img = img.squeeze()
            real_image = img.mul_(127.5).add_(128).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
            # real_image = cv2.cvtColor(real_image, cv2.COLOR_RGB2BGR)
            frames.append(real_image)
        
        imageio.mimwrite(mp4_path, frames, fps=20)

def test_direction():
    convert_img_path = '/path_to/latent_codes/bertha/test_416_bertha.npy'
    direction_path = '/path_to/sketch_collar_direction/bertha_vs_cami_off_shoulder.npy'
    step_list =  np.arange(0,40,0.08)
    mp4_dir = '/path_to/collar_direction_mp4'
    mp4_name = direction_path.split('/')[-1][:-4] + '.mp4'
    mp4_path = os.path.join(mp4_dir, mp4_name)

    args.ckpt = '/path_to/experiment/8_twinnet_fswv1_100000.pt'
    net, opts = setup_model(args.ckpt, device)
    generator = net.decoder
    generator.eval()
    test_direction2(convert_img_path, direction_path, mp4_path, generator, step_list)
    

def get_all_direction():
    base_path = '/path_to/directions/latent_codes2/collar'
    text_num = top_length_test_num
    file_list = os.listdir(base_path)
    print('file_list:  ', file_list)
    direction_name = base_path.split('/')[-1]
    print('1111',direction_name, len(file_list))
    
    for i in range(len(file_list)):
        f1_name = file_list[i]
        f1_path = os.path.join(base_path, f1_name)
        for j in range(0,len(file_list)):
            if j == i:
                continue
            f2_name = file_list[j]
            f2_path = os.path.join(base_path, f2_name)
            collar_pair_direction(f1_path,f2_path, direction_name, text_num)

def convert_image():
    return

def main(args):
    # obtain_latents(args)
    get_all_direction()
    # test_direction()
    pass
    


def get_latents(net, x):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    return codes

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


def get_all_latents(net, data_loader, save_base, n_images=None):
    all_latents = []
    i = 0
    if not os.path.exists(save_base):
        os.makedirs(save_base)

    with torch.no_grad():
        for batch in data_loader:
            if n_images is not None and i > n_images:
                break
            x = batch
            print(x[0])
            fname = x[0][0]+'.npy'
            inputs = x[1].to(device).float()
            latents = get_latents(net, inputs)
            print('code shape: ', latents.size())
            latents = to_numpy(latents)

            save_path = os.path.join(save_base, fname)
            np.save(save_path, latents)
    return 



def setup_data_loader(args, opts):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = args.images_dir if args.images_dir is not None else dataset_args['test_source_root']
    print(f"images path: {images_path}")
    align_function = None
    test_dataset = InferenceDataset(root=images_path,
                                    transform=transforms_dict['transform_test'],
                                    preprocess=align_function,
                                    opts=opts)

    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=2,
                             drop_last=True)

    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)
    return args, data_loader

if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--images_dir", type=str, 
    default='/path_to/images/collar', 
    help="The directory to the images")
    parser.add_argument("--save_dir", type=str, default=None, help="The directory to save.")
    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--edit_attribute", type=str, default='smile', help="The desired attribute")
    parser.add_argument("--edit_degree", type=float, default=0, help="edit degre")
    parser.add_argument("--ckpt", metavar="CHECKPOINT", help="path to generator checkpoint")

    args = parser.parse_args()
    main(args)