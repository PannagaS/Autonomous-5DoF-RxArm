
import cv2
import numpy as np


# def hsv_cv2(hsv_array):
#     hsv_cv2_array = np.array([0, 0, 0], np.uint8)
#     hsv_cv2_array[0] = int(hsv_array[0] * 179 / 359)
#     hsv_cv2_array[1] = int(hsv_array[1] * 255 / 100)
#     hsv_cv2_array[2] = int(hsv_array[2] * 255 / 100)
#     return hsv_cv2_array

# Load the image
image = cv2.imread("All_blocks_2.png")  # Replace with your image file path

# Convert to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges for red, green, and blue
red_lower =  np.array([122, 61, 87], np.uint8) 
red_upper = np.array([179, 255, 255], np.uint8)

green_lower = np.array([36, 52, 43], np.uint8)
green_upper = np.array([85, 255, 156], np.uint8)

blue_lower = np.array([83, 70, 35], np.uint8)
blue_upper = np.array([108, 255, 255], np.uint8)

purple_lower = np.array([108, 81, 4], np.uint8)
purple_upper = np.array([166, 197,  129], np.uint8)

orange_lower = np.array([8, 94, 74], np.uint8)
orange_upper = np.array([12, 255, 212], np.uint8)

yellow_lower = np.array([19, 49, 130], np.uint8)
yellow_upper = np.array([40, 255, 255], np.uint8)

# Create masks for each color
red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
purple_mask = cv2.inRange(hsv_image, purple_lower, purple_upper)
yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
orange_mask = cv2.inRange(hsv_image, orange_lower, orange_upper)

# Apply morphological operations (optional)
kernel = np.ones((5, 5), np.uint8)
red_mask = cv2.dilate(red_mask, kernel)
green_mask = cv2.dilate(green_mask, kernel)
blue_mask = cv2.dilate(blue_mask, kernel)
purple_mask = cv2.dilate(purple_mask, kernel)
orange_mask = cv2.dilate(orange_mask, kernel)
yellow_mask = cv2.dilate(yellow_mask, kernel)


# Find contours for each color
contours, _ = cv2.findContours(red_mask + green_mask + blue_mask + purple_mask+yellow_mask + orange_mask , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours and label objects
for contour in contours:
    area = cv2.contourArea(contour)
    
    if (area > 800) and (area < 3000):  # Adjust the area threshold as needed
        x, y, w, h = cv2.boundingRect(contour)
        
        if np.any(red_mask[y:y+h, x:x+w] > 0):
            label = "Red"
        elif np.any(green_mask[y:y+h, x:x+w] > 0):
            label = "Green"
        elif np.any(blue_mask[y:y+h, x:x+w] > 0):
            label = "Blue"
        elif np.any(purple_mask[y:y+h, x:x+w] > 0):
            label = "Purple"
        elif np.any(yellow_mask[y:y+h, x:x+w] > 0):
            label = "Yellow"
        elif np.any(orange_mask[y:y+h, x:x+w] > 0):
            label = "Orange"
        else:
            label = "Unknown"
        
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Display the result
cv2.imshow("Colored Blocks Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()