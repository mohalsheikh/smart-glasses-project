"""
Handles camera input and preprocessing
Created by Ethan
"""

import cv2 as cv
import src.utils.config as config

class CameraHandler:
    def __init__(
            self,
            camera_index: int = config.CAMERA_INDEX,
            frame_width: int = config.FRAME_WIDTH,
            frame_height: int = config.FRAME_HEIGHT
            ):
        
        # parameters cannot be None and must be of the types specified in the function signature.
        # Our camera index must be a non-negative integer.
        # Our frame width and height must be positive integers.
        # the ifs below ensure that the parameters are valid by raising an error if they aren't.

        # error if camera index is not provided
        if camera_index is None:
            raise ValueError("camera_index must be set.")

        # error if camera index is not a non-negative integer
        if camera_index < 0:
            raise ValueError("camera_index must be a non-negative integer.")
        
        # error if frame width is not provided
        if frame_width is None:
            raise ValueError("frame_width must be set.")
        
        # error if frame width is not a positive integer
        if frame_width <= 0:
            raise ValueError("frame_width must be a positive integer.")
        
        # error if frame height is not provided
        if frame_height is None:
            raise ValueError("frame_height must be set.")
        
        # error if frame height is not a positive integer
        if frame_height <= 0:
            raise ValueError("frame_height must be a positive integer.")

        # initializing video capture object. this should open the camera whose index is specified in config
        self.cap = cv.VideoCapture(camera_index)

        if not self.cap or not self.cap.isOpened(): # if the video capture object was not created successfully...
            raise RuntimeError(f"Could not open camera {camera_index} with OpenCV.") # error

        # setting capture width and height
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

    # attempts to free camera and destroy any open cv windows when the object is deleted.
    def __del__(self):
        # free camera
        try:
            self.cap.release()
        except Exception:
            pass # if release fails, ignore it

        # destroy any open cv windows
        try: 
            cv.destroyAllWindows()
        except Exception:
            pass # if destroyAllWindows fails, ignore it

    def capture_frame(self):
        ret, frame = self.cap.read() # attempt to read a frame from the camera

        if not ret: # if reading the frame failed...
            return None # return None.
        
        return frame # otherwise, return the captured frame
    
    def show_image(self, image, window_name="Camera"):
        cv.imshow(window_name, image) # show the provided frame in a window with the specified name

    def capture_and_show_frame(self, window_name="Camera"):
        frame = self.capture_frame() # call capture_frame to get a frame from the camera.

        if frame is not None: # if capturing the frame was successful...
            self.show_image(frame, window_name) # show the captured frame in a window with the specified name