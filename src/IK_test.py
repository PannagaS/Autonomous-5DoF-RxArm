# import numpy as np
# cos = np.cos
# sin = np.sin
# from scipy.optimize import fsolve

# #IK:
# def IK_equations(theta, xd, yd, zd, pitch):
#     theta1, theta2, theta3 = theta

#     theta1_offset = np.arctan(4)
#     l0 = 103.91 #base height offset
#     l1 = 205.73 #link lengths
#     l2 = 200
#     l3 = 174.15

#     e1 = l1*cos(-theta1+theta1_offset) + l2*cos(-(theta1+theta2)) + l3*cos(-(theta1+theta2+theta3)) - np.sqrt(xd**2 + yd**2)
#     e2 = l0 + l1*sin(-theta1 + theta1_offset) + l2*sin(-(theta1+theta2)) + l3*sin(-(theta1+theta2+theta3)) - zd
#     e3 = theta1 + theta2 + theta3 - pitch #psi = pitch
#     return [e1, e2, e3]

# def IK_numerical(target_position, initial_theta):
#     xd, yd, zd, roll, pitch, roll_flag = target_position

#     theta2, theta3, theta4 = fsolve(IK_equations, initial_theta, args=(xd, yd, zd, pitch))
   
#     theta1 = -np.arctan2(xd, yd)

#     if roll_flag == 0:  #Match roll to
#         theta5 = theta1 + roll  #for now
#     elif roll_flag == 1:    #keep arm parallel to bottom side of board
#         theta5 = theta1
#     elif roll_flag == 2:    #keep parallel tp side of board
#         theta5 = theta1 + np.pi/2

#     joint_angles = np.array([theta1, theta2, theta3, theta4, theta5])

#     print(np.degrees( joint_angles ))

#     if np.max(np.abs( joint_angles )) < 2:
#         return joint_angles
#     else:
#         return IK_numerical(target_position - np.array([0, 0, 0, 0.8*target_position[3], 0, 0]) , initial_theta)

#     # print(np.degrees(np.array([theta1, theta2, theta3, theta4, theta5])))
#     # if np.max(np.abs([theta2, theta3, theta4])) < 2: 
#     #     return np.array([theta1, theta2, theta3, theta4, theta5])

#     # return IK_numerical(target_position-np.array([0, 0, 0, 0, (target_position[3]-np.pi/4)/2, 0]), initial_theta)


# #FK:
# def A_matrix(theta, d, a, alpha):
#         return np.array([
#             [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
#             [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
#             [0, np.sin(alpha), np.cos(alpha), d],
#             [0, 0, 0, 1]
#         ])

# def forward_H(theta_list, d_list, a_list, alpha_list):
#         H1 = A_matrix(theta_list[0], d_list[0], a_list[0], alpha_list[0])
#         H2 = A_matrix(theta_list[1], d_list[1], a_list[1], alpha_list[1])
#         H3 = A_matrix(theta_list[2], d_list[2], a_list[2], alpha_list[2])
#         H4 = A_matrix(theta_list[3], d_list[3], a_list[3], alpha_list[3])
#         H5 = A_matrix(theta_list[4], d_list[4], a_list[4], alpha_list[4])
#         return np.matmul(np.matmul(np.matmul(np.matmul(H1, H2), H3), H4), H5)

# def rotation_matrix_to_euler_xyz(matrix):
#         """
#         Convert a rotation matrix to Euler angles using the XYZ convention.

#         Parameters:
#         - matrix (3x3 ndarray): The rotation matrix.

#         Returns:
#         - ndarray: Euler angles [theta_x, theta_y, theta_z] in radians.
#         """
#         # Check matrix shape
#         if matrix.shape != (3, 3):
#             raise ValueError("The input matrix should be 3x3")

#         # Extract values from the matrix
#         R00 = matrix[0, 0]
#         R01 = matrix[0, 1]
#         R02 = matrix[0, 2]
#         R10 = matrix[1, 0]
#         R11 = matrix[1, 1]
#         R12 = matrix[1, 2]
#         R20 = matrix[2, 0]
#         R21 = matrix[2, 1]
#         R22 = matrix[2, 2]

#         # Compute yaw
#         if np.isclose(R20, -1, rtol=1e-10):
#             theta_x = 0
#             theta_y = np.pi / 2
#             theta_z = np.arctan2(R01, R02)
#         elif np.isclose(R20, 1, rtol=1e-10):
#             theta_x = 0
#             theta_y = -np.pi / 2
#             theta_z = np.arctan2(-R01, -R02)
#         else:
#             theta_y = np.arctan2(-R20, np.sqrt(R21**2 + R22**2))
#             theta_x = np.arctan2(R21, R22)
#             theta_z = np.arctan2(R10, R00)

#         return np.array([theta_x, theta_y, theta_z])

# if __name__ == "__main__":
#     initial_theta = np.array([0.1, 1, 0.2, 1, 0])
#     target_position = np.array([-200, 100, 30, 0, 90, 0])  #x, y, z, roll, pitch, yaw
#     joint_angles = IK_numerical(target_position, initial_theta[1:4])
#     # print(joint_angle)
#     print(np.degrees(joint_angles))

#     #FK test of IK output:
#     dh_params = np.asarray(((np.pi/2., -np.pi/2 + 0.236568, np.pi/2 - 0.236568, -np.pi/2, 0),
#             (103.91, 0, 0, 0, 174.15),
#             (0., 205.73, 200, 0, 0),
#             (-np.pi/2., 0, 0, -np.pi/2, 0)))
#     joint_angle = joint_angles
#     theta_DH = joint_angle + dh_params[0]
#     H = forward_H(theta_DH, dh_params[1], dh_params[2], dh_params[3])
#     Rotation = H[0:3, 0: 3]
#     angle = rotation_matrix_to_euler_xyz(Rotation)
#     target = [H[0, 3], H[1, 3], H[2, 3], angle[0], angle[1], angle[2]]
#     print("\n", target)    

