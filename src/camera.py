#!/usr/bin/env python3

"""!
Class to represent the camera.
"""
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

from math import dist, sqrt
import cv2
import time
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError

#addeds
from numpy.linalg import inv

class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720,1280, 3)).astype(np.uint8)


        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.eye(3)
        self.extrinsic_matrix = np.eye(4)
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])

        #Added for 2.2
        self.apriltag_message = None

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        #print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        image = self.VideoFrame.copy()
        

        original_image = image.copy()

        height, width, _ = image.shape
        
        #create a border mask for exclusion from block detection
        mask_border = np.zeros((height, width), dtype=np.uint8)
        mask_border[28:690, 29:1254] = 255   #contains ymin:ymax, xmin:xmask pixel locations of required mask

        #create another mask to exclude pixels near the stand on the right (noisy)
        # mask_right = np.zeros((height, width), dtype=np.uint8)
        # mask_right[184:563, 1257:1280] = 255 

        #create another mask to exclude pixels near the base of the robot (detects green sometimes)
        mask_base = np.zeros((height, width), dtype=np.uint8)
        mask_base[482:689, 549:731] = 255 
        
    
        #Overall combined mask of 'deadzones' to avoid color detection
        #combined_mask = cv2.bitwise_or(mask_border, mask_right)
        combined_mask = cv2.bitwise_or( mask_base, mask_border)

        # Convert to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        #new hsv (top mask)
        color_ranges = [
            {"name": "Red", "lower": np.array([154, 111, 105]), "upper": np.array([179, 255, 254])},
            {"name": "Orange", "lower": np.array([0, 142, 130]), "upper": np.array([13, 255, 248])},
            {"name": "Yellow", "lower": np.array([21, 67, 153]), "upper": np.array([110, 255, 255])},
            {"name": "Green", "lower": np.array([30, 64, 69]), "upper": np.array([91, 255, 122])},
            {"name": "Blue", "lower": np.array([88, 131, 90]), "upper": np.array([110, 255, 186])},
            {"name": "Purple", "lower": np.array([111, 40, 45]), "upper": np.array([160, 255, 197])}
        ]


        # Create an empty dictionary to store the color areas
        # color_areas = {}
        

        # Iterate over the color ranges
        for color_range in color_ranges:
            # Create a color mask for the current color range
            mask = cv2.inRange(hsv_image, color_range["lower"], color_range["upper"])

            #Apply the border mask
            mask = cv2.bitwise_and(mask, combined_mask)

            # Apply the mask to the image
            # result = cv2.bitwise_and(image, image, mask=mask)

            # Find contours in the masked image
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Iterate through the contours
            for contour in contours:
                # Filter contours by area (adjust the area threshold as needed)
                if cv2.contourArea(contour) > 500:
                    
                    block_area = cv2.contourArea(contour)
                    rotated_rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rotated_rect)
                    box = np.int0(box)
                    angle = rotated_rect[2]  

                    # Draw a contour on the original image
                    cv2.drawContours(original_image, [box], 0, (0, 255, 0), 2)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])


                        # Display the centroid coordinates on the block
                        cv2.putText(original_image, f"({cX}, {cY})", (cX - 20, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(original_image, color_range["name"], (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(original_image, f"Angle: {angle:.2f} degrees", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(original_image, f"Area: {block_area:.2f}", (cX - 20, cY - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        data = [cX, cY, np.radians(angle), block_area ]  
                        dist_threshold = sqrt(block_area)/2           #distance threshold

                        # ##EVENT 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # if self.event1_flag == 1:
                        #     cv2.line(original_image, (34,515), (1242,515), (255,255,255), 1)
                        #     #Outer most check - Event based.
                        #     if(cY < 515):   #EVENT1 - only detect/append if in positive half of board

                        #         # if(500 < block_area <= 1000):
                            
                        #             # empty_flag = 1
                        #             # for c, v in self.blocks["small"].items():
                        #             #     if( len(self.blocks["small"][c]) != 0 ):
                        #             #         empty_flag = 0

                        #             # # if empty_flag == 1:
                        #             #     self.blocks["small"][color_range["name"]].append( data )    #append data
                        #             # else:
                        #             #     distances = []
                        #             #     for color_name, val in self.blocks["small"].items():            #
                        #             #         if len(self.blocks["small"][color_name]) != 0:             #if list of this size/color is non-zero
                    
                        #             #             for block in self.blocks["small"][color_name]:         #iterate over blocks                    
                        #             #                 distances.append( dist( block[0:2], [cX, cY] ) )            #calculate all euclidian distances
                                    
                        #             #     if min(distances) > 2*dist_threshold:                             #if smallest distance ###CHECK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        #             #         self.blocks["small"][color_range ["name"]].append( data )    #append data
                                    
                        #             #old code.
                        #         #     if len(self.blocks["small"][color_range["name"]]) != 0:
                        #         #         distances = []
        
                        #         #         for block in self.blocks["small"][color_range["name"]]:
                        #         #             distances.append( dist( block[0:2], [cX, cY] ) )  
                            
                        #         #         if min(distances) > dist_threshold:
                        #         #             self.blocks["small"][color_range["name"]].append( data )
                            
                        #         #     else:   #if list is empty
                        #         #         self.blocks["small"][color_range["name"]].append( data )    #append data   

        
                        #         # elif(1000 < block_area <= 3000):                                
                            
                        #         #     if len(self.blocks["big"][color_range["name"]]) != 0:
                        #         #         distances = []
        
                        #         #         for block in self.blocks["big"][color_range["name"]]:
                        #         #             distances.append( dist( block[0:2], [cX, cY] ) )  
                            
                        #         #         if min(distances) > dist_threshold:
                        #         #             self.blocks["big"][color_range["name"]].append( data )
                            
                        #         #     else:   #if list is empty
                        #         #         self.blocks["big"][color_range["name"]].append( data )    #append data
                        #         if(500 < block_area <= 1300):   #Small block

                        #             empty_flag = 1
                        #             for c, v in self.blocks["small"].items():
                        #                 if( len(self.blocks["small"][c]) != 0 ):
                        #                     empty_flag = 0


                        #             if empty_flag == 1:
                        #                 self.blocks["small"][color_range["name"]].append( data )    #append data
                        #             else:
                        #                 distances = []
                        #                 for color_name, val in self.blocks["small"].items():            #
                        #                     if len(self.blocks["small"][color_name]) != 0:             #if list of this size/color is non-zero
                    
                        #                         for block in self.blocks["small"][color_name]:         #iterate over blocks                    
                        #                             distances.append( dist( block[0:2], [cX, cY] ) )            #calculate all euclidian distances
                                    
                        #                 if min(distances) > 2*dist_threshold:                             #if smallest distance ###CHECK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        #                     self.blocks["small"][color_range ["name"]].append( data )    #append data

                        #             # else:   #if list is empty
                        #             #     self.blocks["small"][color_range["name"]].append( data )    #append data
        
                        #         elif(1300 < block_area <= 3000):    #big block                             
                            
                        #             if len(self.blocks["big"][color_range["name"]]) != 0:
                        #                 distances = []
        
                        #                 for block in self.blocks["big"][color_range["name"]]:
                        #                     distances.append( dist( block[0:2], [cX, cY] ) )  
                            
                        #                 if min(distances) > dist_threshold:
                        #                     self.blocks["big"][color_range["name"]].append( data )
                            
                        #             else:   #if list is empty
                        #                 self.blocks["big"][color_range["name"]].append( data )    #append data



                        # #####EVENT 2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # msg = self.apriltag_message
                        # if self.event2_flag == 1:
                        #     cv2.rectangle(original_image, (337, 235), (939, 541), (255,255,255), 1)
                        #     #Outer most check - Event based.
                            
                        #     if((338 <= cX <= 939) and (235 <= cY <= 538)): #EVENT2 - detect only within 4 apriltags
                                
                        #         if(500 < block_area <= 1300):   #Small block

                        #             empty_flag = 1
                        #             for c, v in self.blocks["small"].items():
                        #                 if( len(self.blocks["small"][c]) != 0 ):
                        #                     empty_flag = 0


                        #             if empty_flag == 1:
                        #                 self.blocks["small"][color_range["name"]].append( data )    #append data
                        #             else:
                        #                 distances = []
                        #                 for color_name, val in self.blocks["small"].items():            #
                        #                     if len(self.blocks["small"][color_name]) != 0:             #if list of this size/color is non-zero
                    
                        #                         for block in self.blocks["small"][color_name]:         #iterate over blocks                    
                        #                             distances.append( dist( block[0:2], [cX, cY] ) )            #calculate all euclidian distances
                                    
                        #                 if min(distances) > 2*dist_threshold:                             #if smallest distance ###CHECK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        #                     self.blocks["small"][color_range ["name"]].append( data )    #append data

                        #             # else:   #if list is empty
                        #             #     self.blocks["small"][color_range["name"]].append( data )    #append data
        
                        #         elif(1300 < block_area <= 3000):    #big block                             
                            
                        #             if len(self.blocks["big"][color_range["name"]]) != 0:
                        #                 distances = []
        
                        #                 for block in self.blocks["big"][color_range["name"]]:
                        #                     distances.append( dist( block[0:2], [cX, cY] ) )  
                            
                        #                 if min(distances) > dist_threshold:
                        #                     self.blocks["big"][color_range["name"]].append( data )
                            
                        #             else:   #if list is empty
                        #                 self.blocks["big"][color_range["name"]].append( data )    #append data


                        #####EVENT 3 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        msg = self.apriltag_message
                        if self.event3_flag == 1:
                            cv2.line(original_image, (34,515), (1242,515), (255,255,255), 1)
                        #     #Outer most check - Event based.
                            if(cY < 515): 
                            
                               
                                if(500 < block_area <= 1300):   #Small block

                                    empty_flag = 1
                                    for c, v in self.blocks["small"].items():
                                        if( len(self.blocks["small"][c]) != 0 ):
                                            empty_flag = 0


                                    if empty_flag == 1:
                                        self.blocks["small"][color_range["name"]].append( data )    #append data
                                    else:
                                        distances = []
                                        for color_name, val in self.blocks["small"].items():            #
                                            if len(self.blocks["small"][color_name]) != 0:             #if list of this size/color is non-zero
                    
                                                for block in self.blocks["small"][color_name]:         #iterate over blocks                    
                                                    distances.append( dist( block[0:2], [cX, cY] ) )            #calculate all euclidian distances
                                    
                                        if min(distances) > 2*dist_threshold:                             #if smallest distance ###CHECK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                            self.blocks["small"][color_range ["name"]].append( data )    #append data

                                    # else:   #if list is empty
                                    #     self.blocks["small"][color_range["name"]].append( data )    #append data
        
                                elif(1300 < block_area <= 3000):    #big block                             
                            
                                    if len(self.blocks["big"][color_range["name"]]) != 0:
                                        distances = []
        
                                        for block in self.blocks["big"][color_range["name"]]:
                                            distances.append( dist( block[0:2], [cX, cY] ) )  
                            
                                        if min(distances) > dist_threshold:
                                            self.blocks["big"][color_range["name"]].append( data )
                            
                                    else:   #if list is empty
                                        self.blocks["big"][color_range["name"]].append( data )    #append data

        
        self.VideoFrame = original_image
        

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        # self.DepthFrameRaw
        pass

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame and
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """

        self.GridFrame = self.VideoFrame.copy()
 
        if self.calibrate_flag == 1:
            extrinsic_matrix  = self.extrinsic     
        else:
            extrinsic_matrix = self.rough_extrinsic
                                         
        
        # flag = False
        circle_radius = 3
        circle_color = (0,200,0)
        #for x, y in zip(self.grid_points[0].flatten(), self.grid_points[1].flatten()):
        for x in self.grid_x_points:
            for y in self.grid_y_points:
                center_x = x #+ 652 #+self.GridFrame.shape[1]//2
                center_y = y# + 446 #+self.GridFrame.shape[0]//2
                # center = (x, y)
                homogeneous_point =  np.array([[center_x], [center_y],[0], [1]])
                camera_c = extrinsic_matrix @ homogeneous_point
                pixel_c = (1/camera_c[2]) * self.intrinsic@(camera_c[0:3].reshape(-1, 1))


                if self.calibrate_flag == 1:
                    pixel_c = self.H @ (pixel_c)
                    pixel_c = pixel_c/pixel_c[2][0]

                # print(pixel_c)
                cv2.circle(self.GridFrame, (int(pixel_c[0]), int(pixel_c[1])), circle_radius, circle_color, thickness =2)
                #cv2.circle(self.GridFrame, (center_x,center_y), circle_radius, circle_color, thickness =2)

        
            
            # transformed_center = np.linalg.inv(self.intrinsic_matrix)@homogeneous_point
            # cv2.circle(self.VideoFrame, (int(transformed_center[0]), int(transformed_center[1])), circle_radius, circle_color, thickness =2)  # -1 fills the circle
            
            # result = (self.intrinsic_matrix)@homogeneous_point
            # result = np.append(result, 1)
            # transformed_center = extrinsic_matrix @ result
            # transformed_center = transformed_center[0:2]/transformed_center[2]
            
            # cv2.circle(self.VideoFrame, (int(transformed_center[0]), int(transformed_center[1])), circle_radius, circle_color, thickness =2)  # -1 fills the circle
        
        
     
    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """
        modified_image = self.VideoFrame.copy()
        self.apriltag_message = msg
        # print(msg.shape)
        # print(type(msg.detections))
        # print(msg.detections.shape)
        # Write your code here

        #Draw a rectangle using corners 1 and 3 for all detected april tags
        #Draw a circle 
        corner_color = (0, 255, 0)
        center_color = (255, 0, 255)
        circle_radius = 3
        rectangle_thickness = 2
        y_offset = -35
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (255, 0, 0)
        font_scale = 0.6
        font_thickness = 2

        for i in range(len(msg.detections)):
            
            if self.calibrate_flag == 0:
                center =  (np.array( [[int(msg.detections[i].centre.x),     int(msg.detections[i].centre.y),     1]] ) ).T
                corner1 = (np.array( [[int(msg.detections[i].corners[0].x), int(msg.detections[i].corners[0].y), 1]] ) ).T
                # corner2 = (np.array( [[int(msg.detections[i].corners[1].x), int(msg.detections[i].corners[1].y), 1]] ) ).T
                corner3 = (np.array( [[int(msg.detections[i].corners[2].x), int(msg.detections[i].corners[2].y), 1]] ) ).T
            else:
                center =  (self.H)@ (np.array( [[int(msg.detections[i].centre.x),     int(msg.detections[i].centre.y),     1]] ) ).T
                corner1 = (self.H)@ (np.array( [[int(msg.detections[i].corners[0].x), int(msg.detections[i].corners[0].y), 1]] ) ).T
                # corner2 = (self.H)@ (np.array( [[int(msg.detections[i].corners[1].x), int(msg.detections[i].corners[1].y), 1]] ) ).T
                corner3 = (self.H)@ (np.array( [[int(msg.detections[i].corners[2].x), int(msg.detections[i].corners[2].y), 1]] ) ).T

            # print(corner1.shape)
            center =  center/center[2][0]
            corner1 = corner1/corner1[2][0]
            # corner2 = corner2/corner2[2][0]
            corner3 = corner3/corner3[2][0]

            cv2.rectangle( modified_image, (int(corner1[0][0]), int(corner1[1][0])), (int(corner3[0][0]), int(corner3[1][0])) , corner_color, rectangle_thickness)
            cv2.circle( modified_image, (int(center[0][0]), int(center[1][0])), circle_radius, center_color, -1)
            cv2.putText( modified_image, f'ID: {msg.detections[i].id}', (int(corner1[0][0]), int(corner1[1][0])+y_offset), font, font_scale, font_color, font_thickness)


        self.TagImageFrame = modified_image

class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        
        # self.camera.VideoFrame = cv_image
        if self.camera.calibrate_flag == 0:
            self.camera.VideoFrame = cv_image
            # self.camera.blockDetector()
            #print(cv_image.shape)
        else:
            self.camera.VideoFrame = cv2.warpPerspective(cv_image , self.camera.H, (1280,720) )
            self.camera.blockDetector()

            #self.camera.VideoFrame = cv_image




class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')  
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        #print(self.camera.intrinsic_matrix)


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        
        self.camera.DepthFrameRaw = cv_depth
        # if self.camera.calibrate_flag == 0:
        #     self.camera.DepthFrameRaw = cv_depth
        # else:
            # self.camera.DepthFrameRaw = cv2.warpPerspective(cv_depth , self.camera.H, (1280,720) )


        # self.camera.DepthFrameRaw = self.camera.DepthFrameRaw / 2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
        
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                
                grid_frame = self.camera.convertQtGridFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once() # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        
        self.executor.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()