import numpy as np



def linear_stretch(deg_img, decompose_mode=None):

    min_original = np.min(deg_img)
    max_original = np.max(deg_img)
    min_target = 0
    max_target = 1
    new_image = ((deg_img - min_original) * (max_target - min_target) / (max_original - min_original)) + min_target
    
    return new_image