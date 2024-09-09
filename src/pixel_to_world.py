import numpy as np
from numpy.linalg import inv

def pixel_to_world(u, v, Zc):

    u = u
    v = v
    Zc = Zc

    # K = np.array([[892.333, 0.00,   640.667],
    #               [0.00,    894.00, 352.33],
    #               [0.00,    0.00,   1.00]])

    #factory K - intrinsic camera matrix
    K = np.array([[900.71, 0.00,   652.28],
                [0.00,   900.19, 358.35],
                [0.00,   0.00,   1.00]])

    K_inv = inv(K)

    pixel_c = np.array([[u],[v],[1]])

    camera_c = Zc*K_inv@pixel_c

    new_camera_c = np.append(camera_c, [[1]], axis=0)

    # print("\n", new_camera_c)


    # degree = (180)/180 * 1 * np.pi 
    degree = (180 + 9)
    radians = np.radians(degree)
    # print(np.cos(np.radians(degree)))

    #Rough Extrinsic
    # H = np.array([[1, 0, 0,  -7],
    #               [0, np.cos(radians), -np.sin(radians), 180], #320
    #               [0, np.sin(radians), np.cos(radians), 1040], 
    #               [0, 0, 0,  1]])

    #One instance of Extrinsic, 8 ppints, for centres of 8 april tags. LATEST
    # H =np.array([[ 9.99004967e-01, -3.75019609e-02, -2.41387543e-02,  8.27382753e+00],
    #             [-3.34899447e-02, -9.88225526e-01,  1.49294115e-01,  1.90900028e+02],
    #             [-2.94533552e-02, -1.48337157e-01, -9.88498148e-01,  1.04341640e+03],
    #             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    H = np.array([[ 9.99405159e-01, -3.43953915e-02, -2.50693672e-03,  4.46695995e+00],
                [-3.36696753e-02, -9.88877522e-01,  1.44870974e-01,  1.86849448e+02],
                [-7.46194725e-03, -1.44700391e-01, -9.89447379e-01,  1.02890852e+03],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    H_inv = inv(H)
    # print("H_inv in pixel toworld, ",H_inv)
    # print(new_camera_c)
    world_c = H_inv@new_camera_c
    # print("\n", world_c)

    return world_c[0][0], world_c[1][0], world_c[2][0]

if __name__ == '__main__':
    u = 415
    v = 272
    Zc = 997
    world_x, world_y, world_z = pixel_to_world(u, v, Zc)
    print("(%s, %s, %s)"%(round(world_x[0],3), round(world_y[0],3), round(world_z[0],3)) )