

import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk

def update_hsv_range(event):
    global hsv_lower, hsv_upper, image, hsv_image
    
    # Get the lower and upper HSV values from the sliders
    hue_low = hue_low_slider.get()
    hue_high = hue_high_slider.get()
    sat_low = sat_low_slider.get()
    sat_high = sat_high_slider.get()
    val_low = val_low_slider.get()
    val_high = val_high_slider.get()
    
    # Update the HSV range
    hsv_lower = (hue_low, sat_low, val_low)
    hsv_upper = (hue_high, sat_high, val_high)
    
    # Create a mask based on the HSV range
    mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)

    # Display the result
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_rgb)
    result_tk = ImageTk.PhotoImage(result_pil)
    image_label.config(image=result_tk)
    image_label.image = result_tk

# Create the main window
root = tk.Tk()
root.title("HSV Range Tracker")

# Load an image
image_path = filedialog.askopenfilename()
image = cv2.imread(image_path)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create initial HSV range values
hsv_lower = (0, 0, 0)
hsv_upper = (179, 255, 255)

# Create sliders for HSV range
hue_low_slider = tk.Scale(root, from_=0, to=179, label="Hue Low", orient="horizontal", length=300, command=update_hsv_range)
hue_high_slider = tk.Scale(root, from_=0, to=179, label="Hue High", orient="horizontal", length=300, command=update_hsv_range)
sat_low_slider = tk.Scale(root, from_=0, to=255, label="Sat Low", orient="horizontal", length=300, command=update_hsv_range)
sat_high_slider = tk.Scale(root, from_=0, to=255, label="Sat High", orient="horizontal", length=300, command=update_hsv_range)
val_low_slider = tk.Scale(root, from_=0, to=255, label="Val Low", orient="horizontal", length=300, command=update_hsv_range)
val_high_slider = tk.Scale(root, from_=0, to=255, label="Val High", orient="horizontal", length=300, command=update_hsv_range)

# Set initial slider values
hue_low_slider.set(hsv_lower[0])
hue_high_slider.set(hsv_upper[0])
sat_low_slider.set(hsv_lower[1])
sat_high_slider.set(hsv_upper[1])
val_low_slider.set(hsv_lower[2])
val_high_slider.set(hsv_upper[2])

# Display the sliders
hue_low_slider.pack()
hue_high_slider.pack()
sat_low_slider.pack()
sat_high_slider.pack()
val_low_slider.pack()
val_high_slider.pack()

# Create a label to display the image with the applied HSV range
image_label = ttk.Label(root)
image_label.pack()

# Initialize the display with the original image
result_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result_pil = Image.fromarray(result_rgb)
result_tk = ImageTk.PhotoImage(result_pil)
image_label.config(image=result_tk)
image_label.image = result_tk

# Run the main loop
root.mainloop()
