## ---------------------------- ##
## 
## sample_student.py
##
## Example student submission for programming challenge. A few things: 
## 1. Before submitting, change the name of this file to your firstname_lastname.py.
## 2. Be sure not to change the name of the method below, classify.py
## 3. In this challenge, you are only permitted to import numpy and methods from 
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
from util.filters import filter_2d
from util.image import convert_to_grayscale
from util.hough_accumulator import HoughAccumulator

def classify(im):
    '''
    Example submission for coding challenge. 
    
    Args: im (nxmx3) unsigned 8-bit color image 
    Returns: One of three strings: 'brick', 'ball', or 'cylinder'
    
    '''
    def edge(im):
        #Implement Sobel kernels as numpy arrays
        Kx = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])

        Ky = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])
        
        Gx = filter_2d(im,Kx)
        Gy = filter_2d(im,Ky)
        #Compute Gradient Magnitude:
        gm = np.sqrt(Gx**2+Gy**2)
        #Arctan2 works a little better here, allowing us to avoid dividing by zero:
        gd = np.arctan2(Gy, Gx)
        return gm #only sobel for now

        #NMS and chainlinking for canny edge
        gLocalMax=gm
        
        for row in range(1,gm.shape[0]-2):
            for col in range(1,gm.shape[1]-2):
                v= gm[row][col]
                theta = gd[row][col]
                if (theta>=(-np.pi/8.0) and theta<(np.pi/8.0)) and (v<gm[row][col-1] or v<gm[row][col+1]):
                    gLocalMax[row][col]=0
                elif (theta>=(np.pi/8.0) and theta<((3*np.pi)/8.0)) and ((v<gm[row-1][col-1] or v<gm[row+1][col+1])):
                    gLocalMax[row][col]=0
                elif (theta>=((3*np.pi)/8.0) and theta<((5*np.pi)/8.0)) and (v<gm[row-1][col] or v<gm[row+1][col]):
                    gLocalMax[row][col]=0
                elif (theta>=((5*np.pi)/8.0) and theta<((7*np.pi)/8.0)) and (v<gm[row+1][col-1] or v<gm[row-1][col+1]):
                    gLocalMax[row][col]=0
                elif ((theta>=((7*np.pi)/8.0) or theta<((-7*np.pi)/8.0)) and ((v<gm[row][col-1] or v<gm[row][col+1]))):
                    gLocalMax[row][col]=0 
                elif (theta>=((-7*np.pi)/8.0) and theta<((-5*np.pi)/8.0)) and ((v<gm[row-1][col-1] or v<gm[row+1][col+1])):
                    gLocalMax[row][col]=0
                elif (theta>=((-5*np.pi)/8.0) and theta<((-3*np.pi)/8.0)) and ((v<gm[row-1][col] or v<gm[row+1][col])):
                    gLocalMax[row][col]=0
                elif (theta>=((-3*np.pi)/8.0) and theta<((-np.pi)/8.0)) and ((v<gm[row+1][col-1] or v<gm[row-1][col+1])):
                    gLocalMax[row][col]=0    
        #Double threshold
        strongEdges = (gLocalMax > 200)
        #print(gLocalMax.max())
        #Strong has value 2, weak has value 1
        thresholdedEdges = np.array(strongEdges, dtype=np.uint8) + (gLocalMax > 100)

        #Tracing edges with hysteresis	
        #Find weak edge pixels near strong edge pixels
        finalEdges = strongEdges.copy()
        currentPixels = []
        for r in range(1, gm.shape[0]-2):
            for c in range(1, gm.shape[1]-2):
                if thresholdedEdges[r, c] != 1:
                    continue #Not a weak pixel
                #Get 3x3 patch	
                localPatch = thresholdedEdges[r-1:r+2,c-1:c+2]
                #print(localPatch.shape)
                patchMax = localPatch.max()
                if patchMax == 2: 
                    currentPixels.append((r, c))
                    finalEdges[r, c] = 1
        return gm,finalEdges,gLocalMax
    
    im = convert_to_grayscale(im)
#     med = smooth_im(im)
    gm = edge(im)
    # im = im_dil(f,np.array([[0,1,0],[1,1,1],[0,1,0]]))
    # im = im_ero(im,np.array([[0,1,0],[1,1,1],[0,1,0]]))
    edges = gm>200
    y_coords, x_coords = np.where(edges)

    phi_bins = 128
    theta_bins = 128

    accumulator = np.zeros((phi_bins, theta_bins))

    rho_min = -edges.shape[0]
    rho_max = edges.shape[1]

    theta_min = 0
    theta_max = np.pi

    #Compute the rho and theta values for the grids in our accumulator:
    rhos = np.linspace(rho_min, rho_max, accumulator.shape[0])
    thetas = np.linspace(theta_min, theta_max, accumulator.shape[1])

    y_coords_flipped = edges.shape[0] - y_coords
    hough = HoughAccumulator(theta_bins,phi_bins,rho_min,rho_max)
    accumulator = hough.accumulate(x_coords, y_coords_flipped)

#     relative_thresh = 0.80
    max_value = np.max(accumulator)

    if max_value>135:
        return 'brick'
    elif max_value>60:
        return 'cylinder'
    else:
        return 'ball'
