# smart-glasses-project (offline)
Assistive smart glasses project for visually impaired users – real-time object detection, OCR, and text-to-speech.

Run the program through main.py. 

Say the wake word to make it start listening ("vision") and then provide your command. 

Voice commands for detection are in the format of

commandword [object(s)] [direction]

or

commandword [direction] [object(s)]

Direction and/or object(s) do not need to be provided. Only one direction may be provided, and any number of objects. If no directions are provided, the whole scene is described. If no objects are provided, all supported objects are described.

Wake word: vision
Detection command words: detect, read
Directions: left, front, right
Miscellaneous commands: "sleep", "end", "nevermind", or "thanks" to stop listening for commands. "repeat" to repeat the results of the last detection.
Objects: "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "fifty dollar bill", "five dollar bill", "one dollar bill", "ten dollar bill", "twenty dollar bill"

## Notice:
Our project imports some libraries that are not included in the default python installation. If main.py does not run, you may need to install additional libraries. These may include, but may not be limited to...

OpenCV (cv2): pip install opencv-python

Ultralytics: pip install ultralytics

pyttsx3: pip install pyttsx3

numpy: pip install numpy

sounddevice: pip install sounddevice

vosk: pip install vosk

inflect: pip install inflect
