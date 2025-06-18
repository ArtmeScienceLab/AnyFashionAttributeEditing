import os
import numpy as np

base_dir = '/path_to/attribute_editing_results/editing_attribute/svd_directions/'
dataset = 'fp'
flag = '8_rec_twinnet'

if dataset=='fswv1':
    fswv1={
        'attribute':[
            'top_length/short/short_vs_long.npy',
            'dress_length/mini/mini_vs_floorlength.npy',
            'bottom_length/short/short_vs_long.npy',
            'sleeve_length/sleeveless/sleeveless_vs_long.npy',
            'sleeve_shape/regular/regular_vs_bell.npy',
            'collar/r/r_vs_shirt.npy',
            'collar/r/r_vs_v_neck.npy'
        ],
    }
    targ = fswv1

if dataset=='fsmenv3':
    fsmenv3={
        'attribute':[
            'top_length/regular/regular_vs_long.npy',
            'bottom_length/shorts/shorts_vs_long.npy',
            'sleeve_length/sleeveless/sleeveless_vs_long.npy',
            'sleeve_shape/regular/regular_vs_bell.npy',
            'collar/r/r_vs_shirt.npy',
            'collar/r/r_vs_v_neck.npy'
        ],
    }
    targ = fsmenv3

if dataset=='fp':
    fp={
        'attribute':[
            'top_length/short/short_vs_long.npy',
            'dress_length/short/short_vs_long.npy',
            'pants_length/short/short_vs_long.npy',
            'sleeve_length/short/short_vs_long.npy',
            'sleeve_shape/regular/regular_vs_bell.npy',
            'collar/regular/regular_vs_shirt.npy',
            'collar/regular/regular_vs_v.npy'
        ],
    }
    targ = fp


for i in range(len(targ['attribute'])):
    a = targ['attribute'][i]
    attr_path = base_dir + '/' + dataset+'/'+flag+'/'+ a
    attr_a = np.load(attr_path)
    attr_a = np.reshape(attr_a,(18*512))

    # attr_a_length = np.linalg.norm(attr_a)
    
    for j in range(i,len(targ['attribute'])):
        b = targ['attribute'][j]
        attr_path = base_dir + '/' + dataset+'/'+flag+'/'+ b
        attr_b = np.load(attr_path)
        attr_b = np.reshape(attr_b,(18*512))

        # attr_b_length = np.linalg.norm(attr_b)
     
        cos_distance = np.dot(attr_a, attr_b)/(np.linalg.norm(attr_a)*np.linalg.norm(attr_b))
        print('distance between ',a,' and ',b, cos_distance)

