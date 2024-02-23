import numpy as np
import cv2
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

def topleft2perspective(uv, h, w, cx, cy, c):
    '''
    Project from top left to perspective image coordinate 
    
    Parameters
    ----------
    uv : np.array(Nx2), the array of measurements to project
        
    h, w : image height and width
         
    cx, cy, c : corresponding camera calibration parameters
            
    Return
    ------
    xy : np.array(Nx2), the reprojected array 
    '''
    # Your code : convert coordinates

    return xy

#Define function to project from sensor to perspective coordinate system
def perspective2topleft(xy, h, w, cx, cy, c):
    '''
    Project from perspective to top-left image coordinate 
    
    Parameters
    ----------
    xy : np.array(Nx2), the array of measurements to project

    h, w : image height and width

    cx, cy, c : corresponding camera calibration parameters
    
    Return
    ------
    uv : np.array(Nx2), the reprojected array    
    '''
    # Your code : convert coordinates
    
    return uv  

def undistort(var,x_d,y_d,k1,k2,p1,p2):  
    '''
    Define equation (1) and (2) given a complet set of parameter and distorded measurement coordinates
    
    Parameters
    ----------
    var : tupple containing x and y variables. This are the two variables scipy solver will solve for
        
    x_d, y_d : floats, the coordinates of a measurment on the distorded image
        
    k1, k2, p1, p2 : floats, Contrady-Brown distorsion model coefficients
        
    Return
    ------
    
    list of size 2 containing both eqation of the Contrady-Brown model
    '''
    #Unpack variables
    x, y = var
    #Your code : 2 elements list containing equation (1) and (2) depending on x,y

    return

def plot_measurements(img, raw_pts, undist_pts):
    raw_pts_i = raw_pts.astype(int)
    undist_pts_i = undist_pts.astype(int)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    r = (255,0,0)
    g = (0,255,0)
    for i in range(len(raw_pts)):
        cv2.line(img, (raw_pts_i[i]+[-15, 0]), (raw_pts_i[i]+[ 15, 0]), r, 1)
        cv2.line(img, (raw_pts_i[i]+[0, -15]), (raw_pts_i[i]+[0,  15]), r, 1)
        cv2.line(img, (undist_pts_i[i]+[-15, 0]), (undist_pts_i[i]+[ 15, 0]), g, 1)
        cv2.line(img, (undist_pts_i[i]+[0, -15]), (undist_pts_i[i]+[0,  15]), g, 1)
        plt.imshow(img[raw_pts_i[i,1]-150:raw_pts_i[i,1]+150,
                        raw_pts_i[i,0]-150:raw_pts_i[i,0]+150])
        plt.show()

def plot_distortions(img, raw_pts, undist_pts, thickness, out_path, img_id):
    dist = np.linalg.norm(raw_pts - undist_pts, axis=1)
    min_d = np.min(dist)
    max_d = np.max(dist)

    raw_pts_i = raw_pts.astype(int)
    undist_pts_i = undist_pts.astype(int)

    for i in range(len(raw_pts)):
        color = (0,
                 255*(1 -(dist[i]-min_d)/(max_d-min_d))**2,
                 255*((dist[i]-min_d)/(max_d-min_d))**0.5)
        
        cv2.line(img, (raw_pts_i[i]), (undist_pts_i[i]), color, thickness)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20,20))
    plt.imshow(img)



def grid_points(img, k):
    '''
    Generates a regular grid of points from an image in top left coordinates
    
    Parameters
    ----------
    img : cv2.image
        the image from which take a grid
    k : int
        the grid spacing in pixel
            
    Return
    ------
    grid : Nx2 np.array
        the array of the grid, each line contains x,y coordinate of a given node
    '''
    height, width, channels = img.shape
    
    x, y = np.meshgrid(np.arange(0,width, k),
                       np.arange(0,height, k))
    return np.array([x.flatten(), y.flatten()]).T

def main():
    img_id = 1092311568

    im_path = f'raw_data/{img_id}_marked.jpg'
    out_path = 'measurements/'

    img = cv2.imread(im_path)

    marker_path = f'measurements/YOUR_FILE_NAME_HERE.xml'

    cam_IO_path = 'raw_data/cam_param.txt'

    out_path = 'measurements/'
   

if __name__ == "__main__":
    main()