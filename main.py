"""
Main entry point for Smart Glasses Project - FAST VERSION
Created by Mohammed
"""
# Optimize thread usage for CPU-constrained devices
#---Eric
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
#---
import sys
from src.controller import MainController


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