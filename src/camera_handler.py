"""
Handles camera input and preprocessing
"""

import cv2 as cv
import src.utils.config as config

class CameraHandler:
    def __init__(self):
        # initializing video capture object. this should open the camera whose index is specified in config
        self.cap = cv.VideoCapture(config.CAMERA_INDEX)

        if not self.cap or not self.cap.isOpened(): # if the video capture object was not created successfully...
            raise RuntimeError(f"Could not open camera {config.CAMERA_INDEX} with OpenCV.") # error

        # setting capture width and height
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

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