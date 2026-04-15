import os

# Generate speech
os.system('pico2wave -w=voice.wav "Hello, this sounds clearer and smoother."')

# Add 3 seconds of silence at the beginning
os.system('sox voice.wav output.wav pad 3 0')

# Play final audio
os.system('aplay output.wav')
