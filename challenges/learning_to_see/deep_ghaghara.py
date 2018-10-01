## ---------------------------- ##
## 
## sample_student.py
##
## Example student submission for programming challenge. A few things: 
## 1. Before submitting, change the name of this file to your firstname_lastname.py.
## 2. Be sure not to change the name of the method below, count_fingers.py
## 3. In this challenge, you are only permitted to import numpy, and methods from 
##    the util module in this repository. Note that if you make any changes to your local 
##    util module, these won't be reflected in the util module that is imported by the 
##    auto grading algorithm. 
## 4. Anti-plagarism checks will be run on your submission
##
##
## ---------------------------- ##


import numpy as np
#It's kk to import whatever you want from the local util module if you would like:
#from util.X import ... 

def count_fingers(im):
    '''
    Example submission for coding challenge. 
    
    Args: im (nxm) unsigned 8-bit grayscale image 
    Returns: One of three integers: 1, 2, 3
    
    '''
    im = im>50
    T = np.zeros_like(im)
    for col in range(0,im.shape[1],1):
        for row in range(0,im.shape[0],1):
            if (row+9>=im.shape[0] or col+9>=im.shape[1]):
                continue
            cut = im[row:row+9,col:col+9]
            if (cut[4,4] and not cut[0,0] and cut[5,8]):
                T[row+4,col+4]=1    
    top_half = T[:20,:]
    a = top_half.sum(axis=0)
    a = np.diff(a)
    peaks = np.where((a[1:-1] > a[0:-2]) * (a[1:-1] > a[2:]))[0] + 1
    if peaks.shape[0] < 1:
        return 1
    elif peaks.shape[0]>3:
        return 3
    else:
        return peaks.shape[0]