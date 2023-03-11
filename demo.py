## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import sys

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

tracker = cv2.TrackerKCF_create()
boundingBox = (287, 320, 86, 100)

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

counter = 0
# Start streaming
pipeline.start(config)
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
frames = pipeline.wait_for_frames()
#THIS JUST GIVES THE CAMERA A CHANCE TO "WARM UP", IF WE DIDNT DO THIS THE TRACKING FAILIED EVERY TIME
for i in range(20):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
color_frame = frames.get_color_frame()
color_image = np.asarray(color_frame.get_data())
#DRAW A BOX AROUND THE OBJECT YOU WISH TO TRACK ON THAT FIRST FRAME, THEN CLICK SPACEBAR (OR WHATEVER THE ROI CONFIRM BUTTON IS)
bbox = cv2.selectROI(color_image, False)
thing = tracker.init(color_image, bbox)
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
       
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asarray(depth_frame.get_data())
        color_image = np.asarray(color_frame.get_data())
        blank_image = np.zeros((480,90,3), np.uint8)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        blank_image = np.zeros((depth_colormap_dim[0],depth_colormap_dim[1],3), np.uint8)
        
        bl, boundingBox = tracker.update(color_image)
        if bl:
            # Tracking success
            p1 = (int(boundingBox[0]), int(boundingBox[1]))
            p2 = (int(boundingBox[0] + boundingBox[2]), int(boundingBox[1] + boundingBox[3]))
            cv2.rectangle(color_image, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(blank_image, "track Fail", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
            blank_image = cv2.resize(blank_image, dsize=(depth_colormap_dim[1] * 2, depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            cv2.rectangle(blank_image, p1, p2, (255,0,0), 2, 1)
            images = np.vstack((images, blank_image))

    
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        
        cv2.waitKey(1)
        if cv2.waitKey(1) == 27:
            break

finally:

    # Stop streaming
    pipeline.stop()