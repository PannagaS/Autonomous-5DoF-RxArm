import numpy as np
cos = np.cos
sin = np.sin
from scipy.optimize import fsolve


def get_dh_params():
    return

# pox_config_file="rx200_pox.csv"
# M_matrix, S_list = parse_pox_param_file(pox_config_file)

"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm
from scipy.optimize import fsolve

LW_T = 0.16415  # Distance from end effector to wrist (mm)
LE_W = 0.200  # Distance from elbow to wrist (mm)
LS_E = 0.20573  # Distance from shoulder to elbow
GS30 = np.array([[1, 0, 0, 0], [0, 1, 0, 0.05], [0, 0, 1, 0.30391],
                 [0, 0, 0, 1]])  # Homogenous transformation for elbow at home position
GS20 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.10391],
                 [0, 0, 0, 1]])  # Homogenous transformation for shoulder at home position
GS40 = np.array([[1, 0, 0, 0], [0, 1, 0, 0.25], [0, 0, 1, 0.30391],
                 [0, 0, 0, 1]])  # Homogenous transformation for wrist angle at home position
GS50 = np.array([[1, 0, 0, 0], [0, 1, 0, 0.315], [0, 0, 1, 0.30391],
                 [0, 0, 0, 1]])  # Homogenous transformation for wrist rotation at home position


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def IK_equations(theta, xd, yd, zd, pitch):
    theta1, theta2, theta3 = theta

    theta1_offset = np.arctan(4)
    l0 = 103.91 #base height offset
    l1 = 205.73 #link lengths
    l2 = 200
    l3 = 174.15

    e1 = l1*cos(-theta1+theta1_offset) + l2*cos(-(theta1+theta2)) + l3*cos(-(theta1+theta2+theta3)) - np.sqrt(xd**2 + yd**2)
    e2 = l0 + l1*sin(-theta1 + theta1_offset) + l2*sin(-(theta1+theta2)) + l3*sin(-(theta1+theta2+theta3)) - zd
    e3 = theta1 + theta2 + theta3 - pitch #psi = pitch
    return [e1, e2, e3]

def IK_numerical(target_position, initial_theta):
    xd, yd, zd, roll, pitch, roll_flag = target_position

    theta2, theta3, theta4 = fsolve(IK_equations, initial_theta, args=(xd, yd, zd, pitch))
    # theta2, theta3, theta4 = fsolve(IK_equations, [0.1, 0.1, 1], args=(xd, yd, zd, pitch))
   
    theta1 = -np.arctan2(xd, yd)

    if roll_flag == 0:  #Match roll to
        theta5 = theta1 + roll  #for now
    elif roll_flag == 1:    #keep arm parallel to bottom side of board
        theta5 = theta1
    elif roll_flag == 2:    #keep parallel tp side of board
        theta5 = theta1 + np.pi/2
    elif roll_flag == 3:
        theta5 = theta1 - np.pi/2
    elif roll_flag == 4:        #for when pitch is 0 
        theta5 = 0

    joint_angles = np.array([theta1, theta2, theta3, theta4, theta5])

    # print(np.degrees( joint_angles ))

    if np.max(np.abs( joint_angles )) < 3:
        return joint_angles
    else:
        return IK_numerical(target_position - np.array([1, 1, 1, 0.5*target_position[3], 0, 0]) , initial_theta)

    # if np.max(np.abs([theta2, theta3, theta4])) < 2: 
    #     return np.array([theta1, theta2, theta3, theta4, theta5])

    # return IK_numerical(target_position-np.array([0, 0, 0, 0, (target_position[3]-np.pi/4)/2, 0]), initial_theta)