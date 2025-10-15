"""
Main entry point for Smart Glasses Project
Coordinates all components together
Created by Mohammed
"""

from ultralytics import YOLO
from src.controller import MainController

def main():
    controller = MainController()
    controller.run()

if __name__ == "__main__":
    main()
