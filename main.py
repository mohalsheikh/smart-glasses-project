"""
Main entry point for Smart Glasses Project - FAST VERSION
Created by Mohammed
"""

import sys
from src.auto_controller import MainController # change this to auto_controller for the auto version, or manual_controller for the manual version


def main():
    print("\n" + "=" * 60)
    print("⚡ SMART GLASSES - SPEED OPTIMIZED")
    print("=" * 60 + "\n")
    
    try:
        controller = MainController()
        controller.run()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()