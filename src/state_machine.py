"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import cv2
import numpy as np
import rclpy
import apriltag_msgs.msg as msg
import copy
from pixel_to_world_auto import *
from copy import deepcopy
from math import dist, sqrt
class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,          0.0,        0.0],
            [0.75*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.5*-np.pi/2,   -0.5,      -0.3,      np.pi/2,        0.0],
            [0.25*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.0,             0.0,       0.0,          0.0,        0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [0.5*np.pi/2,     0.5,       0.3,     -np.pi/3,        0.0],
            [0.75*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [np.pi/2,         0.5,       0.3,     -np.pi/3,        0.0],
            [0.0,             0.0,       0.0,          0.0,        0.0]]

        self.repeat_waypoints = [np.array([-0.02454369, -1.37444687,  0.88357294,  0.76392245, -0.07823303,
        0.        ]), np.array([-0.42031074,  0.00920388,  0.33900976,  1.29467988, -0.40803891,
        0.        ]), np.array([-0.43258259,  0.14572819,  0.49700978,  1.03083515, -0.41264084,
        0.        ]), np.array([-0.4310486 ,  0.1441942 ,  0.49854377,  1.03083515, -0.41110685,
        1.        ]), np.array([-0.40036899,  0.01687379,  0.15339808,  1.45421386, -0.37582532,
        1.        ]), np.array([-1.2716701 ,  0.11351458,  0.15646605,  1.40666044, -1.23178661,
        1.        ]), np.array([-1.282408  ,  0.19328159,  0.42337871,  1.05384481, -1.25479627,
        1.        ]), np.array([-1.28547597,  0.19174761,  0.41417482,  1.05231082, -1.25479627,
        0.        ]), np.array([-1.23945653,  0.05675729,  0.04141748,  1.53551483, -1.20724297,
        0.        ]), np.array([0.40803891, 0.03528156, 0.29299033, 1.31922352, 0.41110685,
        0.        ]), np.array([0.41264084, 0.15493207, 0.48933989, 1.03083515, 0.42798066,
        0.        ]), np.array([0.42184472, 0.15033013, 0.4878059 , 1.03236914, 0.42798066,
        1.        ]), np.array([0.41570881, 0.03681554, 0.04295146, 1.52017498, 0.42337871,
        1.        ]), np.array([-0.40957287,  0.01380583,  0.32980588,  1.29467988, -0.40343696,
        1.        ]), np.array([-0.41570881,  0.13038836,  0.52462143,  0.98328173, -0.37429133,
        1.        ]), np.array([-0.41417482,  0.13038836,  0.52615541,  0.98174775, -0.38349521,
        0.        ]), np.array([-0.40036899, -0.07516506,  0.1764078 ,  1.44500995, -0.38042724,
        0.        ]), np.array([-1.27627206,  0.0644272 ,  0.2883884 ,  1.28700995, -1.22718465,
        0.        ]), np.array([-1.28854394,  0.20095149,  0.41264084,  1.05384481, -1.25786424,
        1.        ]), np.array([-1.26093221,  0.10737866,  0.02300971,  1.58613622, -1.23638856,
        1.        ]), np.array([0.40190297, 0.0644272 , 0.30372819, 1.2931459 , 0.40803891,
        1.        ]), np.array([0.42031074, 0.14112623, 0.51234961, 0.99555355, 0.42337871,
        1.        ]), np.array([0.4172428 , 0.13805827, 0.51388359, 0.99555355, 0.42337871,
        0.        ]), np.array([ 0.38349521,  0.04755341, -0.07056312,  1.57079637,  0.41110685,
        0.        ])]

        #2.2 - For warping perspective.
        self.camera.calibrate_flag = 0
        self.camera.H = []
        self.camera.april_tag_locations = []
        self.camera.extrinsic = []
        self.camera.intrinsic = np.array([[900.71, 0.00,   652.28],
                                          [0.00,   900.19, 358.35],
                                          [0.00,   0.00,   1.00]])
        
        self.camera.rough_extrinsic = np.array([[ 9.99405159e-01, -3.43953915e-02, -2.50693672e-03,  4.46695995e+00],
         [-3.36696753e-02, -9.88877522e-01,  1.44870974e-01,  1.86849448e+02],
         [-7.46194725e-03, -1.44700391e-01, -9.89447379e-01,  1.02890852e+03],
         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
   
        self.mouse_clicks = []
        self.camera.event1_flag = 1
        self.camera.event2_flag = 1
        self.camera.event3_flag = 1
        self.camera.blocks = {
            "big": {
                "Red": [],
                "Orange": [],
                "Yellow": [],
                "Green": [],
                "Blue": [],
                "Purple": []  },

            "small": {
                "Red": [],
                "Orange": [],
                "Yellow": [],
                "Green": [],
                "Blue": [],
                "Purple": []
            }
        } 
        #initializa a dictionary. Contains empty dictionary of 'big' and 'small' blocks.
        #each dictionary contains a dictionary of lists. Each color is the key has a list of lists. 
        #to access: blocks["big"]["Red"][2][2], for instance. (2 may be the block #3), (1 may be the angle)
        #each list of datavalues for a block is in the form [cX, cY, Angle, Area]

        

        #copy waypoints from "waypoints_trial.txt" for latest successful run

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "capture_waypoint":
            self.capture_waypoint()
        
        if self.next_state == "reset_waypoints":
            self.reset_waypoints()

        if self.next_state == "closegripper":
            self.closegripper()

        if self.next_state == "opengripper":
            self.opengripper()
        
        if self.next_state == "click2move":
            # time.sleep(1)
            self.click2move()
        
        if self.next_state == 'event1':
            self.event1()
        
        if self.next_state == 'event2':
            self.event2()
        
        if self.next_state == 'event3':
            self.event3()


    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    #EVENT 1 HELPER CODES

    def clear_dict(self):
        self.camera.blocks = {
            "big": {
                "Red": [],
                "Orange": [],
                "Yellow": [],
                "Green": [],
                "Blue": [],
                "Purple": []  },

            "small": {
                "Red": [],
                "Orange": [],
                "Yellow": [],
                "Green": [],
                "Blue": [],
                "Purple": []
            }
        } 

    def check_dict(self, copy_block):
        flag = 0
        
        for size,color in copy_block.items():
            for c, coordinates in color.items():
                if len(coordinates) !=0: 
                    print(coordinates)
                    flag = 1
                    break

        return flag

    def move_block1(self, source_pixels, destination_coords, angle, block_height):
        #source pixels is [pixel_x, pixel_y]      --> pixels
        #destination _oords = np.array([x, y, z]) --> world coords
        dest_x, dest_y, dest_z = destination_coords

        x_offset = 5
        y_offset = 2
        z_offset = 5
        move_time = 2
        accel_time = 0.75
        
        #source world coordinates
        world_x, world_y, world_z = pixel_to_world_auto(source_pixels[0], source_pixels[1], self.camera.calibrate_flag, self.camera.H, self.camera.intrinsic, self.camera.extrinsic, self.camera.DepthFrameRaw)
        world_x += x_offset
        world_y += y_offset
        world_z += z_offset


        #move1 - Go to point above source pixels
        target_position = np.array([(world_x), (world_y), (world_z + 100), angle, np.pi/2, 0])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)

        #move2 = Pick up block. Closegripper()
        target_position = np.array([(world_x), (world_y), (world_z - block_height/2 + 20), angle, np.pi/2, 0])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)
        self.closegripper()
        time.sleep(0.5)
        
        #move3 = go back to above block
        target_position = np.array([(world_x), (world_y), (world_z + 100), angle, np.pi/2, 0])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)

        #move4 = go to point above destination
        target_position = np.array([(dest_x), (dest_y), (dest_z + 150), 0, np.pi/2, 1])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)

        #move5 = place at destination. opengripper()
        target_position = np.array([(dest_x), (dest_y), (dest_z + 30), 0, np.pi/2, 1])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)
        self.opengripper()
        time.sleep(0.5)

        #move6 = move to point above destination.
        target_position = np.array([(dest_x), (dest_y), (dest_z + 100), 0, np.pi/2, 1])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)

        print("Move complete!!")

    #move function for event 2
    def move_block2(self, source_pixels, destination_coords, angle, block_height):
        #source pixels is [pixel_x, pixel_y]      --> pixels
        #destination _oords = np.array([x, y, z]) --> world coords
        dest_x, dest_y, dest_z = destination_coords

        x_offset = 0
        y_offset = 0
        z_offset = 0
        move_time = 2
        accel_time = 0.75
        
        #source world coordinates
        world_x, world_y, world_z = pixel_to_world_auto(source_pixels[0], source_pixels[1], self.camera.calibrate_flag, self.camera.H, self.camera.intrinsic, self.camera.extrinsic, self.camera.DepthFrameRaw)
        world_x += x_offset
        world_y += y_offset
        world_z += z_offset


        #move1 - Go to point above source pixels
        target_position = np.array([(world_x), (world_y), (world_z + 100), angle, np.pi/2, 0])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)

        #move2 = Pick up block. Closegripper()
        target_position = np.array([(world_x), (world_y), (world_z - block_height/2 + 20), angle, np.pi/2, 0])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)
        self.closegripper()
        time.sleep(0.5)
        
        #move3 = go back to above block
        target_position = np.array([(world_x), (world_y), (world_z + 100), angle, np.pi/2, 0])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)

        #move4 = go to point above destination
        target_position = np.array([(dest_x), (dest_y), (dest_z + 100), 0, np.pi/2, 1])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)

        #move5 = place at destination. opengripper()
        target_position = np.array([(dest_x), (dest_y), (dest_z + 20), 0, np.pi/2, 1])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)
        self.opengripper()
        time.sleep(0.5)

        #move6 = move to point above destination.
        target_position = np.array([(dest_x), (dest_y), (dest_z + 100), 0, np.pi/2, 1])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)

        print("Move complete!!")
    

        #move function for event 3
    def move_block3(self, source_pixels, destination_coords, angle, block_height, size_flag):
        #source pixels is [pixel_x, pixel_y]      --> pixels
        #destination _oords = np.array([x, y, z]) --> world coords
        dest_x, dest_y, dest_z = destination_coords

        if size_flag == 0: #small block
            roll_flag = 3
        else:
            roll_flag = 2

        x_offset = 10
        y_offset = 0
        z_offset = 10
        move_time = 2
        accel_time = 0.75
        
        #source world coordinates
        world_x, world_y, world_z = pixel_to_world_auto(source_pixels[0], source_pixels[1], self.camera.calibrate_flag, self.camera.H, self.camera.intrinsic, self.camera.extrinsic, self.camera.DepthFrameRaw)
        world_x += x_offset
        world_y += y_offset
        world_z += z_offset

        #Home
        #self.rxarm.set_positions(np.radians(np.array([-1.4, -62.67, 40.43, 90, 2.37])))


        #move1 - Go to point above source pixels
        target_position = np.array([(world_x), (world_y), (250), angle, np.pi/2, 0])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)

        #move1.5 = Pick up block. Closegripper()
        target_position = np.array([(world_x), (world_y), (world_z + 40), angle, np.pi/2, 0])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)
        # self.closegripper()
        time.sleep(0.5)

        #move2 = Pick up block. Closegripper()
        target_position = np.array([(world_x), (world_y), (world_z - block_height/2 + 5), angle, np.pi/2, 0])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)
        self.closegripper()
        time.sleep(0.5)

        #move2.5 = Pick up block. Closegripper()
        target_position = np.array([(world_x), (world_y), (world_z + 40), angle, np.pi/2, 0])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)
        # self.closegripper()
        time.sleep(0.5)
        
        #move3 = go back to above block
        target_position = np.array([(world_x), (world_y), (250), angle,  np.pi/2, 0])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)

        #Home
        #self.rxarm.set_positions(np.radians(np.array([-1.4, -62.67, 40.43, 90, 2.37])))
        

        #move4 = go to point above destination
        target_position = np.array([(dest_x), (dest_y), (200), 0, np.pi/2, 4])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)

        #move4.5 = Pick up block. Closegripper()
        target_position = np.array([(dest_x), (dest_y), (dest_z + 40), angle, np.pi/2, roll_flag])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)
        # self.closegripper()
        time.sleep(0.5)

        #move5 = place at destination. opengripper()
        target_position = np.array([(dest_x), (dest_y), (dest_z + 10), 0, np.pi/2, roll_flag])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)
        self.opengripper()
        time.sleep(0.5)

        #move5.5 = Pick up block. Closegripper()
        target_position = np.array([(dest_x), (dest_y), (dest_z + 40), angle, np.pi/2, roll_flag])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)
        # self.closegripper()
        time.sleep(0.5)

        #move6 = move to point above destination.
        target_position = np.array([(dest_x), (dest_y), (200), 0, np.pi/2, roll_flag])
        init_angle = self.rxarm.get_positions()
        target_angle = self.rxarm.ik_execute(target_position, init_angle)
        self.rxarm.set_positions(target_angle)
        time.sleep(move_time)

        print("Move complete!!")
    


    #EVENT 1 - Competition
    def event1(self):
        self.current_state = "event1"
        self.status_message = " executing event1 "

        print(self.camera.blocks)
        self.camera.event1_flag = 0

        dest_big = [[150, -50, 0], [225,-50,0], [300,-50,0], [150, -120, 0], [255, -120, 0], [300, -120, 0], [150, -75, 37], [250, -75, 37] ]
        dest_small =  [[-150, -50, 0], [-250,-50,0], [-300,-50,0], [-150, -120, 0], [-250, -120, 0], [-300, -120, 0], [-150, -75, 37], [-250, -75, 37] ]
        dbig_i = 0
        dsmall_i = 0

        big_block_height = 37
        small_block_height = 25

        #Pass1:
        #Big blocks
        blocks_copy = deepcopy(self.camera.blocks)

        while(True):
            for color in blocks_copy["small"]:
                for block in blocks_copy["small"][color]:
                    
                    self.move_block1( block[0:2], dest_small[dsmall_i], block[2],  small_block_height)
                    dsmall_i += 1
            
            for color in blocks_copy["big"]:
                for block in blocks_copy["big"][color]:
                    
                    self.move_block1( block[0:2], dest_big[dbig_i], block[2],  big_block_height)
                    dbig_i += 1
        

            self.clear_dict()
            self.camera.event1_flag = 1
            time.sleep(1)
            blocks_copy = deepcopy(self.camera.blocks)
            self.camera.event1_flag = 0
            print(blocks_copy)
            d_flag = self.check_dict(blocks_copy)

            if d_flag == 0:
                break


            #Next pass on updated copy of 
        self.camera.event1_flag = 1            
        print("EVENT1 COMPLETE!!!")
        self.next_state = "idle"



    #EVENT 2

    def event2(self):
        self.current_state = "event2"
        self.status_message = " executing event2 "

        print(self.camera.blocks)
        self.camera.event2_flag = 0

        dest_big = [[-250,-25,0], [-250,-25,40], [-250,-25,80], [-250,-25,120] ]
        dest_small =  [[250,-25,0], [250,-25,30], [250,-25,60], [250,-25,90] ]
        dbig_i = 0
        dsmall_i = 0
        # dest = [[-250,-25,0],[250,-25,0]] #4 april tages + stacking on them
        # d_i = 0

        big_block_height = 37
        small_block_height = 25

        #Pass1:
        #Big blocks
        blocks_copy = deepcopy(self.camera.blocks)

        while(True):
            
            for color in blocks_copy["small"]:
                for block in blocks_copy["small"][color]:
                    
                    self.move_block2( block[0:2], dest_small[dsmall_i], block[2],  small_block_height)
                    dsmall_i += 1

            for color in blocks_copy["big"]:
                for block in blocks_copy["big"][color]:
                    
                    self.move_block2( block[0:2], dest_big[dbig_i], block[2],  big_block_height)
                    dbig_i += 1
            
            self.clear_dict()
            self.camera.event2_flag = 1
            time.sleep(1)
            blocks_copy = deepcopy(self.camera.blocks)
            self.camera.event2_flag = 0
            print(blocks_copy)
            d_flag = self.check_dict(blocks_copy)

            if d_flag == 0:
                break

            #Next pass on updated copy of 
        # self.camera.event2_flag = 1            
        print("EVENT2 COMPLETE!!!")

        self.next_state = "idle"



    #EVENT 3 - Competition
    def event3(self):
        self.current_state = "event3"
        self.status_message = " executing event3 "

        print(self.camera.blocks)
        self.camera.event3_flag = 0

        big_block_height = 37
        small_block_height = 25

        #Pass1:
        #Big blocks
        blocks_copy = deepcopy(self.camera.blocks)
        color_dest_big = {
            "Red": [135, -100, 5],
            "Orange": [185, -100, 5],
            "Yellow":[235, -100, 5],
            "Green":[285, -100, 5],
            "Blue":[335, -100, 5],
            "Purple" : [385, -100, 5]
        }

        color_dest_small = {
            "Red": [-135, -100, 5],
            "Orange": [-175, -100, 5],
            "Yellow":[-215, -100, 5],
            "Green":[-255, -100, 5],
            "Blue":[-295, -100, 5],
            "Purple" : [-335, -100, 5]
        }

        while(True):
            for color in blocks_copy["small"]:

                for block in blocks_copy["small"][color]:
                    
                    print(color_dest_small[color])
                    self.move_block3( block[0:2], color_dest_small[color], block[2],  small_block_height, 0)
                    # dsmall_i += 1
            
            for color in blocks_copy["big"]:
                for block in blocks_copy["big"][color]:
                    
                    self.move_block3( block[0:2], color_dest_big[color], block[2],  big_block_height, 1)
                    # dbig_i += 1
        

            self.clear_dict()
            self.camera.event3_flag = 1
            time.sleep(1)
            blocks_copy = deepcopy(self.camera.blocks)
            self.camera.event3_flag = 0
            print(blocks_copy)
            d_flag = self.check_dict(blocks_copy)

            if d_flag == 0:
                break


            #Next pass on updated copy of 
        self.camera.event1_flag = 1            
        print("EVENT3 COMPLETE!!!")
        self.next_state = "idle"


    #added for checkpoint 3
    def click2move(self):

        self.current_state = "click2move"
        self.status_message = " click 2 move state "
    
        # pixel_x = self.camera.last_click[0]
        # pixel_y = self.camera.last_click[1]
        big_block_z_offset = 34/2 #mm
        # pixel_z = self.camera.DepthFrameRaw[pixel_y][pixel_x] + big_block_z_offset        

        self.rxarm.set_moving_time(2.)
        self.rxarm.set_accel_time(0.5)
        zero_click = np.array([0, 0]) 

        x_offset = 0
        y_offset = 0
        z_offset = 20

        if not np.array_equal(self.camera.last_click, zero_click):
            if len(self.mouse_clicks) == 1:

                if not np.array_equal( self.camera.last_click, self.mouse_clicks[0] ):
                # if not (self.camera.last_click == self.mouse_clicks[-1] ).all():
                    self.mouse_clicks.append(np.array([self.camera.last_click[0], self.camera.last_click[1]]))

                    world_x, world_y, world_z = pixel_to_world_auto(self.camera.last_click[0], self.camera.last_click[1], self.camera.calibrate_flag, self.camera.H, self.camera.intrinsic, self.camera.extrinsic, self.camera.DepthFrameRaw)
                    world_z = world_z - big_block_z_offset
    
                    #home to above block
                    # target_position = np.array([(world_x+15)/1000, (world_y-10)/1000, (world_z+80 + 35)/1000, 0, np.pi/2, 0])
                    target_position = np.array([(world_x+x_offset), (world_y-y_offset), (world_z+z_offset+100), 0, np.pi/2, 0])
                    init_angle = self.rxarm.get_positions()
                    # target_angle = self.rxarm.ik_execute(target_position)
                    target_angle = self.rxarm.ik_execute(target_position, init_angle)
                    self.rxarm.set_positions(target_angle)
                    time.sleep(2)
                    # self.closegripper()
    
                    # above block to block above block
                    # target_position = np.array([(world_x+15)/1000, (world_y-10)/1000, (world_z+20 + 35)/1000, 0, np.pi/2, 0])
                    target_position = np.array([(world_x+x_offset), (world_y-y_offset), (world_z+z_offset+50), 0, np.pi/2, 0])
                    init_angle = self.rxarm.get_positions()
                    # target_angle = self.rxarm.ik_execute(target_position)
                    target_angle = self.rxarm.ik_execute(target_position, init_angle)
                    self.rxarm.set_positions(target_angle)
                    time.sleep(2)
                    self.opengripper()
                    time.sleep(1)
    
                    # target_position = np.array([(world_x+15)/1000, (world_y-10)/1000, (world_z+80 + 35)/1000, 0, np.pi/2, 0])
                    target_position = np.array([(world_x+x_offset), (world_y-y_offset), (world_z+z_offset+100), 0, np.pi/2, 0])
                    init_angle = self.rxarm.get_positions()
                    # target_angle = self.rxarm.ik_execute(target_position)
                    target_angle = self.rxarm.ik_execute(target_position, init_angle)
                    self.rxarm.set_positions(target_angle)
                    time.sleep(1)

            else:
                self.mouse_clicks.append(np.array([self.camera.last_click[0], self.camera.last_click[1]]))
                
                world_x, world_y, world_z = pixel_to_world_auto(self.camera.last_click[0], self.camera.last_click[1], self.camera.calibrate_flag, self.camera.H, self.camera.intrinsic, self.camera.extrinsic, self.camera.DepthFrameRaw)
                world_z = world_z - big_block_z_offset

                #home to above block
                # target_position = np.array([(world_x+15)/1000, (world_y-10)/1000, (world_z+80)/1000, 0, np.pi/2, 0])
                target_position = np.array([(world_x+x_offset), (world_y-y_offset), (world_z+z_offset+50), 0, np.pi/2, 0])
                init_angle = self.rxarm.get_positions()
                # target_angle = self.rxarm.ik_execute(target_position)
                target_angle = self.rxarm.ik_execute(target_position, init_angle)
                print(target_angle)
                self.rxarm.set_positions(target_angle)
                time.sleep(2)
                # self.closegripper()

                # above block to block above block
                # target_position = np.array([(world_x+15)/1000, (world_y-10)/1000, (world_z+20)/1000, 0, np.pi/2, 0])
                target_position = np.array([(world_x+x_offset), (world_y-y_offset), (world_z+z_offset), 0, np.pi/2, 0])
                init_angle = self.rxarm.get_positions()
                # target_angle = self.rxarm.ik_execute(target_position)
                target_angle = self.rxarm.ik_execute(target_position, init_angle)
                self.rxarm.set_positions(target_angle)
                time.sleep(2)
                self.closegripper()
                time.sleep(2)

                # target_position = np.array([(world_x+15)/1000, (world_y-10)/1000, (world_z+80)/1000, 0, np.pi/2, 0])
                target_position = np.array([(world_x+x_offset), (world_y-y_offset), (world_z+z_offset+50), 0, np.pi/2, 0])
                init_angle = self.rxarm.get_positions()
                # target_angle = self.rxarm.ik_execute(target_position)
                target_angle = self.rxarm.ik_execute(target_position, init_angle)
                self.rxarm.set_positions(target_angle)
                time.sleep(1)

        if len(self.mouse_clicks) == 2:
            self.mouse_clicks = []
            print("cleared")
            self.next_state = "idle"
        else:
            # print(len(self.mouse_clicks))
            self.next_state = "click2move"



    #Added for 1.3
    def reset_waypoints(self):
        self.status_message = "resetting waypoints"

        self.repeat_waypoints = []                  #Resets to empty list. 
        
        self.next_state = "idle"

    #Added for 1.3
    def capture_waypoint(self):
        self.status_message = "capturing waypoint"

        current_point = self.rxarm.get_positions()       #Get position, typecast to list from nparray
        

        if self.rxarm.gripper_state:
            current_point = np.append(current_point, 1)
            self.repeat_waypoints.append(current_point)           #Appends list with current waypoint
        
        else:
            current_point = np.append(current_point, 0)
            self.repeat_waypoints.append(current_point)           #Appends list with current waypoint

        print("\n", self.repeat_waypoints)
        print("gripper state = ", self.rxarm.gripper_state)

        with open(r'/home/student_am/armlab-wolverine-8/src/waypoints.txt', 'w') as fp: 
            fp.write(str(list(self.repeat_waypoints)))
        self.next_state = "idle"

    def closegripper(self):
        self.status_message = "closed "
        self.rxarm.gripper.grasp()
        self.rxarm.gripper_state = True
        print(self.rxarm.gripper_state)
        self.next_state = "idle"

    def opengripper(self):
        self.status_message = "open"
        self.rxarm.gripper.release()
        self.rxarm.gripper_state = False
        print(self.rxarm.gripper_state)
        self.next_state = "idle"

    #Modified for 1.2
    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """

        for iterations in range(2):
            for i in range(len(self.repeat_waypoints)):
                if i == 0:
                    angluar_displacement = self.repeat_waypoints[i][0:5] - np.array([-np.pi/2, -0.5, -0.3, 0.0, 0.0])
                else:
                    angluar_displacement = self.repeat_waypoints[i][0:5] - self.repeat_waypoints[i-1][0:5]
                
                print(f'waypoint number {i}')

                index = np.argmax(np.abs(angluar_displacement))
                max_angle = np.abs(angluar_displacement[index])
                print("max angle =", max_angle)

                if max_angle >= 0.8:
                    constant_time = 0.9
                elif (max_angle < 0.8) and (max_angle >= 0.4):
                    constant_time = 1.5
                else:
                    constant_time = 10
                    max_angle = 0.1  #sets a lower bound on max_angle - for the extremely low values of max_angle
                    print("HERE------------")

                moving_time = constant_time * (max_angle)
                print("moving time = ", moving_time)
                
                print(type(self.repeat_waypoints[i][0:5]))
                print(self.repeat_waypoints[i][0:5].shape)
                
                self.repeat_waypoints[i][0:5]
                test_angle = np.array([200, 200, 30, 0, np.pi/2., 0])
                init_angle = np.zeros(5)
                target_angle = self.rxarm.ik_execute(test_angle, init_angle)
                target = np.round(target_angle, 3)
                print(type(target_angle))
                print(target_angle.shape)
                print(target_angle)
                self.rxarm.set_positions(self.repeat_waypoints[i][0:5])

                time.sleep(0.5)
                if self.repeat_waypoints[i][-1] == True:
                    self.closegripper()
                else:
                    self.opengripper()
                
                time.sleep(0.5)
            # print(f"Iteration {iteration} complete -----------")

        print(" 1 ")
        self.rxarm.set_moving_time(4.)
        self.rxarm.set_accel_time(1.)

        target_position = np.array([250/1000., 250/1000., 150/1000., 1.56, 1.57, -1.57])

        init_angle = self.rxarm.get_positions()

        target_angle = self.rxarm.ik_execute(target_position)
        print(np.degrees(target_angle))
        self.rxarm.set_positions(target_angle)
        print("2")
        time.sleep(0.5)
        print("\n CONTINUING EXECUTION")

            #NEED TO ACCOUNT FOR ESTOP - Transition to estop from this state.
        # time.sleep(1)
        
        self.next_state = "idle"
        

    #Modified for 2.2
    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """TODO Perform camera calibration routine here"""

        #print(f'OLD = {self.camera.rough_extrinsic}')

        self.status_message = "Auto Calibration Starting"

        msg = self.camera.apriltag_message

        tag_centres = []
        image_points = []
        for i in range(8):
            # self.april_tag_locations.append([tag])
            centre =  ( int(msg.detections[i].centre.x), int(msg.detections[i].centre.y) )
            corner1 = ( int(msg.detections[i].corners[0].x), int(msg.detections[i].corners[0].y) )
            corner2 = ( int(msg.detections[i].corners[1].x), int(msg.detections[i].corners[1].y) )
            corner3 = ( int(msg.detections[i].corners[2].x), int(msg.detections[i].corners[2].y) )
            corner4 = ( int(msg.detections[i].corners[3].x), int(msg.detections[i].corners[3].y) ) 

            tag_centres.append( centre )
            # image_points.extend([centre, corner1, corner2, corner3, corner4])
            image_points.extend([centre])
            self.camera.april_tag_locations.extend([[centre, corner1, corner2, corner3, corner4]])
#       
        #Calculate Homography
        src_points = np.float32([ tag_centres[0], tag_centres[1], tag_centres[2], tag_centres[3] ])               
        # dst_points = np.float32([ [328,225],          [949,225],      [949, 550],    [328,550]       ][::-1])       #Image Corners
        dst_points = np.float32([ [338,235],          [939,235],      [939, 540],    [338,540]       ][::-1])       #Image Corners

        self.camera.H = cv2.getPerspectiveTransform( src_points, dst_points )   #Save matrix

        #Calculate Extrinsic
        #u, v coordinates of the 4 april tags from msg - (all centers and no corners) = 8 points 
        image_points = np.array(image_points,dtype = np.float32)

        #world coordinates of the 4 april tags centers and no corners, in the same order = 8 points
        object_points = np.array([(-250, -25, 0), #(-262, -37, 0),       (-236, -37, 0),       (-236, -12, 0),       (-262, -12, 0),   #center1, corner1-4
                                (250, -25, 0),    #(237, -37, 0),        (262, -37, 0),        (262, -12, 0),        (237, -12, 0),      #center2, corner1-4
                                (250, 275, 0),    #(234, 263, 0),        (258, 263, 0),        (258, 288, 0),        (234, 288, 0),      #center3, corner1-4
                                (-250, 275, 0),   #(-260, 262, 0),       (-235, 262, 0),       (-235, 286, 0),       (-260, 286, 0),     #center4, corner1-4

                                (375, 400, 154), #(-437.5, 387.5, 245), (-412.5, 387.5, 245), (-412.5, 412.5, 245), (-437.5, 412.5, 245), #center5, corner1-4 -top-left, h=245
                                (-375, 100, 96),   #(412.5, 387.5, 154),   (437.5, 387.5, 154),   (437.5, 412.5, 154),   (412.5, 412.5, 154),  #center6, corner1-4 - top-right, h=154

                                (-375, 400, 154),#(-437.5, -112.5, 154),(-412.5, -112.5, 154),(-412.5, -87.5, 154), (-437.5, -87.5, 154),  #center7, corner1-4 -bottom-left, h=154
                                (375, 100, 66)  #(412.5, -112.5, 93),  (437.5, -112.5, 93),  (437.5, -87.5, 93),   (412.5, -87.5, 93),  #center8, corner1-4 -middle, h=93
                                ], dtype = np.float32)

        # dist_coefficients = np.zeros((4,1)) # Not using this
        dist_coefficients = np.array([[0.1490122675895691], [-0.5096240639686584], [-0.0006352968048304319], [0.0005230441456660628], [0.47986456751823425]])
        pnpFlag, rvec, tvec = cv2.solvePnP( object_points, image_points, self.camera.intrinsic , dist_coefficients)
        
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        # print(f'norm = {np.linalg.norm(rotation_matrix[:, 0])}')

        #Build the Extrinsic into a 4x4 matrix
        extrinsic = np.hstack((rotation_matrix, tvec))
        self.camera.extrinsic = np.vstack((extrinsic, [0, 0, 0, 1]))
        # print(f'NEW = {self.camera.extrinsic}')

        self.camera.extrinsic = self.camera.rough_extrinsic
        # print(f'NEW2 = {self.camera.extrinsic}')



        self.camera.calibrate_flag = 1
        # print("Calibration Complete") #only a terminal check, can remove

        with open(r'/home/student_am/armlab-wolverine-8/src/extrinsic_matrix.txt', 'w') as fp: 
            fp.write(str(self.camera.extrinsic))

        self.status_message = "Calibration - Completed Calibration"

    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        time.sleep(1)

    #competition
    def remove_lower_blocks(self):
        self.current_state = "remove_lower_blocks"
        self.status_message = "RXArm Initialized!"

        d = self.camera.blocks
        bottom_blocks = []
        centroid_threshold = 60 #width of tiny block

        for color1,coords1 in d["small"].items():
            for item1 in coords1:
                
                for color2,coords2 in d["small"].items():
                    for item2 in coords2:
                        if color2 != color1:

                            if dist(item1[0:2], item2[0:2]) < centroid_threshold:
                                if item1[3] > item2[3]:
                                    bottom_blocks.append([color2, item2])
                                else:
                                    bottom_blocks.append([color1, item1])
        

        print("==================")
        print(bottom_blocks, "\n\n")

        unique_dict ={}
        for item in bottom_blocks:
            key = item[0]
            if key not in unique_dict:
                unique_dict[key] = item

        # Create a new list with unique items
        unique_list = list(unique_dict.values())

        for item in unique_list:
            self.camera.blocks["small"][item[0]].remove(item[1])


        print('unique_list =', unique_list)



        self.next_state = "idle"

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.camera.event1_flag = 0 #######################################added
        self.camera.event2_flag = 0 #######################################added
        self.camera.event3_flag = 0 #######################################added


        self.remove_lower_blocks()


        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            time.sleep(5)
        self.next_state = "idle"

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)