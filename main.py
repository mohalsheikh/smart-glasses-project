"""
Main entry point for Smart Glasses Project
Coordinates all components together
"""

from src.controller import MainController

def main():
    controller = MainController()
    controller.run()

if __name__ == "__main__":
    main()
