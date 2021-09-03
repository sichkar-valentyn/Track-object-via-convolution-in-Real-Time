

"""
Course:  Convolutional Neural Networks for Image Classification

Section-1
Quick Win #5: Convolution in Real Time by camera

Description:
Track object via convolution in Real Time
Calculate object centre and visualize tracker line

File: track_object_via_convolution_in_real_time.py
"""


# Algorithm:
# --> Defining 1-channeled common 3x3 filter (kernel) for edge detection
# --> Preparing OpenCV windows to be shown
# --> Reading frames from camera in the loop
#     --> Initializing Conv2D layer for GRAY input
#     --> Implementing 2D convolution to the captured frame
#     --> Finding the biggest contour
#         --> Drawing bounding box, label, and prepare tracker line
#         --> Cutting detected fragment
#         --> Showing OpenCV windows, visualizing tracker line
#     --> Calculating FPS rate
#
# Result: OpenCV windows with results


# Importing needed libraries
import cv2
import numpy as np

import tensorflow as tf

from timeit import default_timer as timer

from collections import deque


"""
Start of:
Additional libraries to overcome possible CUDA <--> Tensorflow issues
"""
# Discussion #1
# https://stackoverflow.com/questions/43147983/could-
# not-create-cudnn-handle-cudnn-status-internal-error

# Discussion #2
# https://stackoverflow.com/questions/47068709/your-
# cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

"""
End of:
Additional libraries to overcome possible CUDA <--> Tensorflow issues
"""


"""
Start of:
Defining 1-channeled common 3x3 filter (kernel) for edge detection
"""

# Sobel filter to detect vertical changes on image
f1 = np.array([[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]])

"""
End of:
Defining 1-channeled common 3x3 filter (kernel) for edge detection
"""


"""
Start of:
Preparing OpenCV windows to be shown
"""

# Giving names to the windows
# Specifying that windows are resizable

# Window to show current view from camera in Real Time
cv2.namedWindow('Current view', cv2.WINDOW_NORMAL)

# Window to show found contour
cv2.namedWindow('Contour', cv2.WINDOW_NORMAL)

# Window to show cut fragment
cv2.namedWindow('Cut fragment', cv2.WINDOW_NORMAL)

# Window to show tracker line
cv2.namedWindow('Tracker line', cv2.WINDOW_NORMAL)

"""
End of:
Preparing OpenCV windows to be shown
"""


"""
Start of:
Reading frames from camera in the loop
"""

# Defining 'VideoCapture' object
# to read stream video from camera
# Index of the built-in camera is usually 0
# Try to select other cameras by passing 1, 2, 3, etc.
camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture('videos/bird.mp4')


# Preparing variables for spatial dimensions of the captured frames
h, w = None, None


# Getting version of OpenCV that is currently used
# Converting string into the list by dot as separator and getting first number
v = cv2.__version__.split('.')[0]


# Creating image with black background
temp = np.zeros((720, 1280, 3), np.uint8)


# Initializing deque object for center points of the detected object
points = deque(maxlen=50)


# Defining counter for FPS (Frames Per Second)
counter = 0

# Starting timer for FPS
# Getting current time point in seconds
fps_start = timer()


# Defining loop to catch frames
while True:
    # Capturing frames one-by-one from camera
    _, frame_bgr = camera.read()

    # If the frame was not retrieved
    # e.g.: at the end of the video,
    # then we break the loop
    if not _:
        break

    """
    Start of:
    Initializing Conv2D layer for GRAY input
    """

    # Getting spatial dimension of the frame
    # We do it only once from the very beginning
    # All other frames have the same dimensions
    if w is None or h is None:
        # Slicing from tuple only first two elements
        (h, w) = frame_bgr.shape[:2]

        # If you're using environment for GPU, there might be an issue like:
        '''Failed to get convolution algorithm.
        This is probably because cuDNN failed to initialize'''
        # In this case, try to switch to the environment for CPU usage only
        # instead of GPU

        # Initializing Conv2D layer for GRAY input
        # We do it only once from the very beginning
        layer = tf.keras.layers.Conv2D(filters=1,
                                       kernel_size=(3, 3),
                                       strides=1,
                                       padding='same',
                                       activation='relu',
                                       input_shape=(h, w, 1),
                                       use_bias=False,
                                       kernel_initializer=
                                       tf.keras.initializers.constant(f1))

    """
    End of:
    Initializing Conv2D layer for GRAY input
    """

    """
    Start of:
    Implementing 2D convolution to the captured frame
    """

    # Converting captured frame to GRAY by OpenCV function
    frame_GRAY = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Reshaping frame to get following: (batch size, rows, columns, channels)
    x_input_GRAY = frame_GRAY.reshape(1, h, w, 1).astype(np.float32)

    # Passing GRAY input to the initialized Conv2D layer
    # Calculating time spent for 2D convolution
    start = timer()
    output = layer(x_input_GRAY)
    end = timer()

    # Slicing from the output just filter
    # Converting output Tensor into Numpy array
    output = np.array(output[0, :, :, 0])

    # To exclude values that are less than 0 and more than 255,
    # Numpy function 'clip' is applied
    # It keeps values of Numpy array in the given range
    # And it replaces non-needed values with boundary numbers
    output = np.clip(output, 0, 255).astype(np.uint8)

    """
    End of:
    Implementing 2D convolution to the captured frame
    """

    """
    Start of:
    Finding the biggest contour
    """

    # Applying dilation
    # Morphological filter that increases areas of foreground pixels
    # dilated = cv2.dilate(output, None, iterations=3)
    dilated = output

    # Finding contours
    # (!) Different versions of OpenCV returns different number of parameters
    # when using function cv2.findContours()

    # In OpenCV version 3 function cv2.findContours() returns three parameters:
    # modified image, found contours and hierarchy
    # All found contours from current frame are stored in the list
    # Each individual contour is a Numpy array of(x, y) coordinates
    # of the boundary points of the object
    # We are interested only in contours

    # Checking if OpenCV version 3 is used
    if v == '3':
        _, contours, _ = cv2.findContours(dilated,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_NONE)

    # In OpenCV version 4 function cv2.findContours() returns two parameters:
    # found contours and hierarchy
    # All found contours from current frame are stored in the list
    # Each individual contour is a Numpy array of(x, y) coordinates
    # of the boundary points of the object
    # We are interested only in contours

    # Checking if OpenCV version 4 is used
    else:
        contours, _ = cv2.findContours(dilated,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)

    # Finding the biggest Contour by sorting from biggest to smallest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    """
    Start of:
    Drawing bounding box, label, and prepare tracker line
    """

    # Extracting rectangle coordinates around biggest contour if any was found
    if contours:
        # Function cv2.boundingRect() is used to get an approximate rectangle
        # around the region of interest in binary image after contour was found
        (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])

        # Drawing bounding box on the current BGR frame
        cv2.rectangle(frame_bgr, (x_min, y_min),
                      (x_min + box_width, y_min + box_height),
                      (0, 255, 0), 3)

        # Preparing text for the label
        label = 'Person'

        # Putting text with label on the current BGR frame
        cv2.putText(frame_bgr, label, (x_min - 5, y_min - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # Getting current center coordinates of the bounding box
        center = (int(x_min + box_width / 2), int(y_min + box_height / 2))

        # Adding current pont to the queue
        points.appendleft(center)

        """
        End of:
        Drawing bounding box, label, and prepare tracker line
        """

        """
        Start of:
        Cutting detected fragment
        """

        # Cutting detected fragment from BGR frame
        cut_fragment_bgr = frame_bgr[y_min + int(box_height * 0.1):
                                     y_min + box_height - int(box_height * 0.1),
                                     x_min + int(box_width * 0.1):
                                     x_min + box_width - int(box_width * 0.1)]

        """
        End of:
        Cutting detected fragment
        """

        """
        Start of:
        Showing OpenCV windows, visualizing tracker line
        """
        
        # Showing current view from camera in Real Time
        # Pay attention! 'cv2.imshow' takes images in BGR format
        cv2.imshow('Current view', frame_bgr)

        # Showing found contour
        cv2.imshow('Contour', output)

        # Showing cut fragment
        cv2.imshow('Cut fragment', cut_fragment_bgr)

        # Changing background to BGR(230, 161, 0)
        # B = 230, G = 161, R = 0
        temp[:, :, 0] = 230
        temp[:, :, 1] = 161
        temp[:, :, 2] = 0

        # Visualizing tracker line
        for i in range(1, len(points)):
            # If no points collected yet
            if points[i - 1] is None or points[i] is None:
                continue

            # Draw the line between points
            cv2.line(temp, points[i - 1], points[i], (0, 255, 0), 3)

        # Adding text with center coordinates of the bounding box
        cv2.putText(temp, 'X: {0}'.format(center[0]),
                    (50, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255),
                    4, cv2.LINE_AA)
        cv2.putText(temp, 'Y: {0}'.format(center[1]),
                    (50, 200), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255),
                    4, cv2.LINE_AA)

        # Showing tracker line
        cv2.imshow('Tracker line', temp)

        """
        End of:
        Showing OpenCV windows, visualizing tracker line
        """

    # If no contour is found, showing OpenCV windows with information
    else:
        # Showing current view from camera in Real Time
        # Pay attention! 'cv2.imshow' takes images in BGR format
        cv2.imshow('Current view', frame_bgr)

        # Changing background to BGR(230, 161, 0)
        # B = 230, G = 161, R = 0
        temp[:, :, 0] = 230
        temp[:, :, 1] = 161
        temp[:, :, 2] = 0

        # Adding text with information
        cv2.putText(temp, 'No contour', (100, 450),
                    cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 4, cv2.LINE_AA)

        # Showing information in prepared OpenCV windows
        cv2.imshow('Contour', temp)
        cv2.imshow('Cut fragment', temp)
        cv2.imshow('Tracker line', temp)

    """
    End of:
    Finding the biggest contour
    """

    """
    Start of:
    Calculating FPS
    """

    # Increasing counter for FPS
    counter += 1

    # Stopping timer for FPS
    # Getting current time point in seconds
    fps_stop = timer()

    # Checking if timer reached 1 second
    # Comparing
    if fps_stop - fps_start >= 1.0:
        # Showing FPS rate
        print('FPS rate is: ', counter)

        # Reset FPS counter
        counter = 0

        # Restart timer for FPS
        # Getting current time point in seconds
        fps_start = timer()

    """
    End of:
    Calculating FPS
    """

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

"""
End of:
Reading frames from camera in the loop
"""


# Releasing camera
camera.release()

# Destroying all opened OpenCV windows
cv2.destroyAllWindows()


"""
Some comments
OpenCV function 'cv2.findContours'
More details and examples are here:
https://docs.opencv.org/4.0.0/d4/d73/tutorial_py_contours_begin.html


Function 'cv2.putText' adds text to images.
More details and examples are here:
print(help(cv2.putText))
https://docs.opencv.org/master/dc/da5/tutorial_py_drawing_functions.html


Function 'cv2.dilate' increases the boundaries of regions of foreground pixels.
More details and examples are here:
print(help(cv2.dilate))
https://docs.opencv.org/master/db/df6/tutorial_erosion_dilatation.html
https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html
#ga4ff0f3318642c4f469d0e11f242f3b6c


Python deque object is a generalization of stacks and queues.
More details and examples are here:
from collections import deque
print(help(deque))
https://docs.python.org/3/library/collections.html#collections.deque

"""
