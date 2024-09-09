"""!
Implements the RXArm class.

The RXArm class contains:

* last feedback from joints
* functions to command the joints
* functions to get feedback from joints
* functions to do FK and IK
* A function to read the RXArm config file

You will upgrade some functions and also implement others according to the comments given in the code.
"""
import numpy as np
from functools import partial
from kinematics import get_dh_params, IK_numerical
import time
import csv
import sys, os

from builtins import super
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from resource.config_parse import parse_dh_param_file
from sensor_msgs.msg import JointState
import rclpy

sys.path.append('../../interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/interbotix_xs_modules/xs_robot') 
from arm import InterbotixManipulatorXS
from mr_descriptions import ModernRoboticsDescription as mrd

"""
TODO: Implement the missing functions and add anything you need to support them
"""
""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


def _ensure_initialized(func):
    """!
    @brief      Decorator to skip the function if the RXArm is not initialized.

    @param      func  The function to wrap

    @return     The wraped function
    """
    def func_out(self, *args, **kwargs):
        if self.initialized:
            return func(self, *args, **kwargs)
        else:
            print('WARNING: Trying to use the RXArm before initialized')

    return func_out


class RXArm(InterbotixManipulatorXS):
    """!
    @brief      This class describes a RXArm wrapper class for the rx200
    """
    def __init__(self, dh_config_file=None):
        """!
        @brief      Constructs a new instance.

                    Starts the RXArm run thread but does not initialize the Joints. Call RXArm.initialize to initialize the
                    Joints.

        @param      dh_config_file  The configuration file that defines the DH parameters for the robot
        """
        super().__init__(robot_model="rx200")
        self.joint_names = self.arm.group_info.joint_names
        self.num_joints = 5
        # Gripper
        self.gripper_state = False
        # State
        self.initialized = False
        # Cmd
        self.position_cmd = None
        self.moving_time = 2.0
        self.accel_time = 0.5
        # Feedback
        self.position_fb = None
        self.velocity_fb = None
        self.effort_fb = None
        # DH Params
        self.dh_params = get_dh_params()
        self.dh_params = np.asarray(((np.pi/2., -np.pi/2 + 0.2455, np.pi/2 - 0.2455, -np.pi/2, 0),
            (103.91, 0, 0, 0, 174.15),
            (0., 205.73, 200, 0, 0),
            (-np.pi/2., 0, 0, -np.pi/2, 0)))
        # self.dh_config_file = dh_config_file
        # if (dh_config_file is not None):
        #     self.dh_params = RXArm.parse_dh_param_file(dh_config_file)
        #POX params
        self.M_matrix = []
        self.S_list = []

    def initialize(self):
        """!
        @brief      Initializes the RXArm from given configuration file.

                    Initializes the Joints and serial port

        @return     True is succes False otherwise
        """
        self.initialized = False
        # Wait for other threads to finish with the RXArm instead of locking every single call
        time.sleep(0.25)
        """ Commanded Values """
        self.position = [0.0] * self.num_joints  # radians
        """ Feedback Values """
        self.position_fb = [0.0] * self.num_joints  # radians
        self.velocity_fb = [0.0] * self.num_joints  # 0 to 1 ???
        self.effort_fb = [0.0] * self.num_joints  # -1 to 1
        self.S_list = [[0,0,1,0,0,0],[-1,0,0,0,-0.10391,0],[-1,0,0,0,-0.30391,0.050],[-1,0,0,0,-0.30391,0.250],[0,1,0,-0.30391,0,0]] # List of screw vectors (J1-J5)
        # Reset estop and initialized
        self.estop = False
        self.enable_torque()
        self.moving_time = 2.0
        self.accel_time = 0.5
        self.arm.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=False)
        self.gripper.release()
        self.initialized = True
        return self.initialized

    def sleep(self):
        self.moving_time = 2.0
        self.accel_time = 1.0
        self.arm.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=True)
        self.arm.go_to_sleep_pose(moving_time=self.moving_time,
                              accel_time=self.accel_time,
                              blocking=False)
        self.initialized = False

    def set_positions(self, joint_positions):
        """!
         @brief      Sets the positions.

         @param      joint_angles  The joint angles
         """
        self.arm.set_joint_positions(joint_positions,
                                 moving_time=self.moving_time,
                                 accel_time=self.accel_time,
                                 blocking=False)

    def set_moving_time(self, moving_time):
        self.moving_time = moving_time

    def set_accel_time(self, accel_time):
        self.accel_time = accel_time

    def disable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.core.robot_set_motor_registers('group', 'all', 'Torque_Enable', 0)

    def enable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.core.robot_set_motor_registers('group', 'all', 'Torque_Enable', 1)

    def get_positions(self):
        """!
        @brief      Gets the positions.

        @return     The positions.
        """
        return self.position_fb

    def get_velocities(self):
        """!
        @brief      Gets the velocities.

        @return     The velocities.
        """
        return self.velocity_fb

    def get_efforts(self):
        """!
        @brief      Gets the loads.

        @return     The loads.
        """
        return self.effort_fb


#   @_ensure_initialized

    def A_matrix(self, theta, d, a, alpha):
        return np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])


    def forward_H(self, theta_list, d_list, a_list, alpha_list):
        H1 = self.A_matrix(theta_list[0], d_list[0], a_list[0], alpha_list[0])
        H2 = self.A_matrix(theta_list[1], d_list[1], a_list[1], alpha_list[1])
        H3 = self.A_matrix(theta_list[2], d_list[2], a_list[2], alpha_list[2])
        H4 = self.A_matrix(theta_list[3], d_list[3], a_list[3], alpha_list[3])
        H5 = self.A_matrix(theta_list[4], d_list[4], a_list[4], alpha_list[4])
        return np.matmul(np.matmul(np.matmul(np.matmul(H1, H2), H3), H4), H5)

    def rotation_matrix_to_euler_xyz(self, matrix):
        """
        Convert a rotation matrix to Euler angles using the XYZ convention.

        Parameters:
        - matrix (3x3 ndarray): The rotation matrix.

        Returns:
        - ndarray: Euler angles [theta_x, theta_y, theta_z] in radians.
        """
        # Check matrix shape
        if matrix.shape != (3, 3):
            raise ValueError("The input matrix should be 3x3")

        # Extract values from the matrix
        R00 = matrix[0, 0]
        R01 = matrix[0, 1]
        R02 = matrix[0, 2]
        R10 = matrix[1, 0]
        R11 = matrix[1, 1]
        R12 = matrix[1, 2]
        R20 = matrix[2, 0]
        R21 = matrix[2, 1]
        R22 = matrix[2, 2]

        # Compute yaw
        if np.isclose(R20, -1, rtol=1e-10):
            theta_x = 0
            theta_y = np.pi / 2
            theta_z = np.arctan2(R01, R02)
        elif np.isclose(R20, 1, rtol=1e-10):
            theta_x = 0
            theta_y = -np.pi / 2
            theta_z = np.arctan2(-R01, -R02)
        else:
            theta_y = np.arctan2(-R20, np.sqrt(R21**2 + R22**2))
            theta_x = np.arctan2(R21, R22)
            theta_z = np.arctan2(R10, R00)

        return np.array([theta_x, theta_y, theta_z])

    def get_ee_pose(self):
        """!
        @brief      TODO Get the EE pose.

        @return     The EE pose as [x, y, z, phi] or as needed.
        """
        joint_angle = self.get_positions()
        # print(joint_angle)
        # print(self.dh_params[0])
        theta_DH = joint_angle + self.dh_params[0]
        # print(theta_DH)
        H = self.forward_H(theta_DH, self.dh_params[1], self.dh_params[2], self.dh_params[3])
        Rotation = H[0:3, 0: 3]
        angle = self.rotation_matrix_to_euler_xyz(Rotation)
        return [H[0, 3], H[1, 3], H[2, 3], angle[0], angle[1], angle[2]]

    def get_ee_pose_IK(self, joint_angle):
        """!
        @brief      TODO Get the EE pose.

        @return     The EE pose as [x, y, z, phi] or as needed.
        """
        # print(joint_angle)
        # print(self.dh_params[0])
        theta_DH = joint_angle + self.dh_params[0]
        # print(theta_DH)
        H = self.forward_H(theta_DH, self.dh_params[1], self.dh_params[2], self.dh_params[3])
        Rotation = H[0:3, 0: 3]
        angle = self.rotation_matrix_to_euler_xyz(Rotation)
        return [H[0, 3], H[1, 3], H[2, 3], angle[0], angle[1], angle[2]]

    @_ensure_initialized
    def get_wrist_pose(self):
        """!
        @brief      TODO Get the wrist pose.

        @return     The wrist pose as [x, y, z, phi] or as needed.
        """
        return [0, 0, 0, 0]

    def parse_pox_param_file(self):
        """!
        @brief      TODO Parse a PoX config file

        @return     0 if file was parsed, -1 otherwise 
        """
        return -1

    def parse_dh_param_file(self):
        print("Parsing DH config file...")
        dh_params = parse_dh_param_file(self.dh_config_file)
        print("DH config file parse exit.")
        return dh_params

    def get_dh_parameters(self):
        """!
        @brief      Gets the dh parameters.

        @return     The dh parameters.
        """
        return self.dh_params

    ## FSOLVE MOETHOD - IK
    def ik_execute(self, target, initial_guess):
        theta = IK_numerical(target,initial_guess[1:4])
        return theta


class RXArmThread(QThread):
    """!
    @brief      This class describes a RXArm thread.
    """
    updateJointReadout = pyqtSignal(list)
    updateEndEffectorReadout = pyqtSignal(list)

    def __init__(self, rxarm, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      RXArm  The RXArm
        @param      parent  The parent
        @details    TODO: set any additional initial parameters (like PID gains) here
        """
        QThread.__init__(self, parent=parent)
        self.rxarm = rxarm
        self.node = rclpy.create_node('rxarm_thread')
        self.subscription = self.node.create_subscription(
            JointState,
            '/rx200/joint_states',
            self.callback,
            10
        )
        self.subscription  # prevent unused variable warning
        rclpy.spin_once(self.node, timeout_sec=0.5)

    def callback(self, data):
        self.rxarm.position_fb = np.asarray(data.position)[0:5]
        self.rxarm.velocity_fb = np.asarray(data.velocity)[0:5]
        self.rxarm.effort_fb = np.asarray(data.effort)[0:5]
        self.updateJointReadout.emit(self.rxarm.position_fb.tolist())
        self.updateEndEffectorReadout.emit(self.rxarm.get_ee_pose())
        #for name in self.rxarm.joint_names:
        #    print("{0} gains: {1}".format(name, self.rxarm.get_motor_pid_params(name)))
        if (__name__ == '__main__'):
            print(self.rxarm.position_fb)

    def run(self):
        """!
        @brief      Updates the RXArm Joints at a set rate if the RXArm is initialized.
        """
        while True:
            rclpy.spin_once(self.node) 
            time.sleep(0.02)


if __name__ == '__main__':
    rclpy.init() # for test
    rxarm = RXArm()
    print(rxarm.joint_names)
    armThread = RXArmThread(rxarm)
    armThread.start()
    try:
        joint_positions = [-1.0, 0.5, 0.5, 0, 1.57]
        rxarm.initialize()

        rxarm.arm.go_to_home_pose()
        rxarm.set_gripper_pressure(0.5)
        rxarm.set_joint_positions(joint_positions,
                                  moving_time=2.0,
                                  accel_time=0.5,
                                  blocking=True)
        rxarm.gripper.grasp()
        rxarm.arm.go_to_home_pose()
        rxarm.gripper.release()
        rxarm.sleep()

    except KeyboardInterrupt:
        print("Shutting down")

    rclpy.shutdown()