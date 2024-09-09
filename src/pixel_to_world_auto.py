import numpy as np
from numpy.linalg import inv
import cv2

def untransform_pixels(tu, tv, homography):
    #converts transformed pixels to untransformed pixels.
    r_pixels = inv(homography) @ np.array( [[tu], [tv], [1]] )
    u = int(r_pixels[0][0]/(r_pixels[2][0]))
    v = int(r_pixels[1][0]/(r_pixels[2][0]))
    return u, v

theta = np.radians(0.94)   
rotation_matrix = np.array([[1, 0,              0,            0],
                            [0, np.cos(theta), -np.sin(theta), 0],
                            [0, np.sin(theta),  np.cos(theta), 0],
                            [0, 0,              0,             1] ])

def pixel_to_world_auto(u, v, flag, homography, intrinsic, extrinsic, DepthFrameRaw):

    #Set u,v based on if transform has been applied
    u, v = untransform_pixels(u, v, homography)
    
    Zc = DepthFrameRaw[v][u]

        
    #factory K - intrinsic camera matrix
    # K = np.array([[900.71, 0.00,   652.28],
    #               [0.00,   900.19, 358.35],
    #               [0.00,   0.00,   1.00]])
    # K_inv = inv(K)
    K_inv = inv(intrinsic)

    #pixel coordinates (untransformed)
    pixel_c = np.array([[u],[v],[1]])

    camera_c = Zc*(K_inv@pixel_c)
    new_camera_c = np.append(camera_c, [[1]], axis=0) 

    # with open(r'/home/student_am/armlab-wolverine-8/src/extrinsic_matrix.txt', 'w') as fp: 
    #         fp.write(str(Extrinsic))
    
    E_inv = inv(extrinsic)

    world_c = E_inv@new_camera_c

    #rotation matrix about x to offset skew.
    
    # rotation_matrix  = 1 
    


    rotated_world_c = rotation_matrix @ world_c

    return rotated_world_c[0][0], rotated_world_c[1][0], rotated_world_c[2][0]
