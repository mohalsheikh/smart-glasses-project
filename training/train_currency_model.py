#!/usr/bin/env python3
"""
VisionAssist Currency Detection Model Training Script
======================================================
Created by Mohammed for CBU Capstone Project

This script trains a YOLOv8 model to detect US currency denominations.
Run this in the university lab with GPU access.

Usage:
    python train_currency_model.py --download    # First time: download dataset
    python train_currency_model.py --train       # Train the model
    python train_currency_model.py --test        # Test on sample images
    python train_currency_model.py --export      # Export for deployment
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

# Check for required packages
def check_dependencies():
    """Check and install required packages."""
    required = {
        'ultralytics': 'ultralytics',
        'roboflow': 'roboflow',
        'torch': 'torch torchvision --index-url https://download.pytorch.org/whl/cu118',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("=" * 60)
        print("MISSING DEPENDENCIES - Run these commands first:")
        print("=" * 60)
        for pkg in missing:
            print(f"  pip install {pkg}")
        print("=" * 60)
        sys.exit(1)

check_dependencies()

import torch
from ultralytics import YOLO


# ============================================================================
# CONFIGURATION - Modify these settings as needed
# ============================================================================

CONFIG = {
    # Roboflow settings (create free account at roboflow.com)
    'ROBOFLOW_API_KEY': 'YOUR_API_KEY_HERE',  # Replace with your key
    'ROBOFLOW_WORKSPACE': 'alex-hyams-cosqx',
    'ROBOFLOW_PROJECT': 'dollar-bill-detection',
    'ROBOFLOW_VERSION': 24,
    
    # Training settings
    'MODEL_SIZE': 'yolov8m.pt',      # Options: yolov8n.pt (fast), yolov8s.pt, yolov8m.pt (balanced), yolov8l.pt, yolov8x.pt (accurate)
    'EPOCHS': 100,                    # More epochs = better (up to a point)
    'BATCH_SIZE': 16,                 # Reduce if you get CUDA out of memory
    'IMAGE_SIZE': 640,                # Standard YOLO input size
    'PATIENCE': 20,                   # Early stopping patience
    
    # Paths
    'DATA_DIR': './datasets',
    'OUTPUT_DIR': './runs',
    'FINAL_MODEL_DIR': './models',
}


def print_gpu_info():
    """Print GPU information."""
    print("\n" + "=" * 60)
    print("GPU INFORMATION")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA is available!")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️  CUDA is NOT available - training will be SLOW on CPU")
        print("   Make sure you're on a lab computer with GPU!")
    
    print("=" * 60 + "\n")


def download_dataset():
    """Download the currency dataset from Roboflow."""
    from roboflow import Roboflow
    
    print("\n" + "=" * 60)
    print("DOWNLOADING DATASET FROM ROBOFLOW")
    print("=" * 60)
    
    api_key = CONFIG['ROBOFLOW_API_KEY']
    
    if api_key == 'YOUR_API_KEY_HERE':
        print("\n⚠️  You need to set your Roboflow API key!")
        print("\nSteps to get your API key:")
        print("1. Go to https://roboflow.com and create a FREE account")
        print("2. Go to Settings → API Key")
        print("3. Copy your API key")
        print("4. Edit this file and replace 'YOUR_API_KEY_HERE' with your key")
        print("\nOr run with: python train_currency_model.py --download --api-key YOUR_KEY")
        return None
    
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(CONFIG['ROBOFLOW_WORKSPACE']).project(CONFIG['ROBOFLOW_PROJECT'])
        
        print(f"\nDownloading: {CONFIG['ROBOFLOW_PROJECT']} v{CONFIG['ROBOFLOW_VERSION']}")
        print("This may take a few minutes...\n")
        
        dataset = project.version(CONFIG['ROBOFLOW_VERSION']).download(
            "yolov8",
            location=CONFIG['DATA_DIR']
        )
        
        print(f"\n✅ Dataset downloaded to: {CONFIG['DATA_DIR']}")
        print(f"   Dataset location: {dataset.location}")
        
        return dataset.location
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("- Check your API key is correct")
        print("- Check your internet connection")
        print("- Make sure you have a Roboflow account")
        return None


def find_data_yaml():
    """Find the data.yaml file in the dataset directory."""
    data_dir = Path(CONFIG['DATA_DIR'])
    
    # Look for data.yaml in common locations
    possible_paths = [
        data_dir / 'data.yaml',
        data_dir / 'dollar-bill-detection-24' / 'data.yaml',
        data_dir / CONFIG['ROBOFLOW_PROJECT'] / 'data.yaml',
    ]
    
    # Also search recursively
    for yaml_file in data_dir.rglob('data.yaml'):
        possible_paths.append(yaml_file)
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    print(f"❌ Could not find data.yaml in {data_dir}")
    print("   Please run with --download first")
    return None


def train_model(resume=False):
    """Train the YOLOv8 model."""
    print("\n" + "=" * 60)
    print("TRAINING CURRENCY DETECTION MODEL")
    print("=" * 60)
    
    print_gpu_info()
    
    # Find data.yaml
    data_yaml = find_data_yaml()
    if not data_yaml:
        print("Please download the dataset first: python train_currency_model.py --download")
        return None
    
    print(f"Using dataset config: {data_yaml}")
    
    # Load model
    if resume and Path(CONFIG['OUTPUT_DIR']).exists():
        # Find the latest run to resume
        runs = sorted(Path(CONFIG['OUTPUT_DIR']).glob('*/weights/last.pt'))
        if runs:
            model_path = str(runs[-1])
            print(f"Resuming from: {model_path}")
            model = YOLO(model_path)
        else:
            print(f"No checkpoint found, starting fresh with {CONFIG['MODEL_SIZE']}")
            model = YOLO(CONFIG['MODEL_SIZE'])
    else:
        print(f"Loading base model: {CONFIG['MODEL_SIZE']}")
        model = YOLO(CONFIG['MODEL_SIZE'])
    
    # Training configuration
    train_args = {
        'data': data_yaml,
        'epochs': CONFIG['EPOCHS'],
        'imgsz': CONFIG['IMAGE_SIZE'],
        'batch': CONFIG['BATCH_SIZE'],
        'patience': CONFIG['PATIENCE'],
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 8,
        'project': CONFIG['OUTPUT_DIR'],
        'name': 'currency_detector',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True,
        
        # Augmentation settings for robustness
        'augment': True,
        'hsv_h': 0.015,      # Hue augmentation
        'hsv_s': 0.7,        # Saturation augmentation  
        'hsv_v': 0.4,        # Value/brightness augmentation
        'degrees': 15.0,     # Rotation augmentation
        'translate': 0.1,    # Translation augmentation
        'scale': 0.5,        # Scale augmentation
        'shear': 5.0,        # Shear augmentation
        'perspective': 0.0005,
        'flipud': 0.0,       # No vertical flip (bills have orientation)
        'fliplr': 0.5,       # Horizontal flip is OK
        'mosaic': 1.0,       # Mosaic augmentation
        'mixup': 0.1,        # Mixup augmentation
        
        # Save settings
        'save': True,
        'save_period': 10,   # Save checkpoint every 10 epochs
        'plots': True,       # Generate training plots
    }
    
    print("\nTraining configuration:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING - This will take a while...")
    print("=" * 60 + "\n")
    
    # Train!
    try:
        results = model.train(**train_args)
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        
        # Print results
        print(f"\nResults saved to: {CONFIG['OUTPUT_DIR']}/currency_detector/")
        print("\nKey metrics:")
        if hasattr(results, 'box'):
            print(f"  mAP50: {results.box.map50:.4f}")
            print(f"  mAP50-95: {results.box.map:.4f}")
        
        return model
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\nTroubleshooting:")
        print("- If CUDA out of memory: reduce BATCH_SIZE in CONFIG")
        print("- If dataset not found: run --download first")
        raise


def validate_model():
    """Validate the trained model."""
    print("\n" + "=" * 60)
    print("VALIDATING MODEL")
    print("=" * 60)
    
    # Find best model
    best_model = Path(CONFIG['OUTPUT_DIR']) / 'currency_detector' / 'weights' / 'best.pt'
    
    if not best_model.exists():
        print(f"❌ No trained model found at {best_model}")
        print("   Please train the model first: python train_currency_model.py --train")
        return None
    
    model = YOLO(str(best_model))
    
    # Find data.yaml
    data_yaml = find_data_yaml()
    if not data_yaml:
        return None
    
    # Validate
    metrics = model.val(data=data_yaml)
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics


def export_model():
    """Export the model for deployment."""
    print("\n" + "=" * 60)
    print("EXPORTING MODEL FOR DEPLOYMENT")
    print("=" * 60)
    
    # Find best model
    best_model = Path(CONFIG['OUTPUT_DIR']) / 'currency_detector' / 'weights' / 'best.pt'
    
    if not best_model.exists():
        print(f"❌ No trained model found at {best_model}")
        return None
    
    # Create output directory
    output_dir = Path(CONFIG['FINAL_MODEL_DIR'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(str(best_model))
    
    # Copy the PyTorch model
    final_pt = output_dir / 'currency_detector.pt'
    shutil.copy(best_model, final_pt)
    print(f"✅ PyTorch model: {final_pt}")
    
    # Export to ONNX (cross-platform)
    print("\nExporting to ONNX format...")
    model.export(format='onnx', imgsz=CONFIG['IMAGE_SIZE'])
    onnx_src = best_model.with_suffix('.onnx')
    if onnx_src.exists():
        onnx_dst = output_dir / 'currency_detector.onnx'
        shutil.move(str(onnx_src), str(onnx_dst))
        print(f"✅ ONNX model: {onnx_dst}")
    
    # Export to TorchScript (for mobile/embedded)
    print("\nExporting to TorchScript format...")
    model.export(format='torchscript', imgsz=CONFIG['IMAGE_SIZE'])
    ts_src = best_model.with_suffix('.torchscript')
    if ts_src.exists():
        ts_dst = output_dir / 'currency_detector.torchscript'
        shutil.move(str(ts_src), str(ts_dst))
        print(f"✅ TorchScript model: {ts_dst}")
    
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE!")
    print("=" * 60)
    print(f"\nModels saved to: {output_dir}/")
    print("\nTo use in your project, copy 'currency_detector.pt' to your")
    print("VisionAssist models/ directory and update the config.")
    
    return output_dir


def test_inference(image_path=None):
    """Test the model on sample images."""
    print("\n" + "=" * 60)
    print("TESTING MODEL INFERENCE")
    print("=" * 60)
    
    # Find best model
    model_paths = [
        Path(CONFIG['FINAL_MODEL_DIR']) / 'currency_detector.pt',
        Path(CONFIG['OUTPUT_DIR']) / 'currency_detector' / 'weights' / 'best.pt',
    ]
    
    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            break
    
    if not model_path:
        print("❌ No trained model found")
        return None
    
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    if image_path and Path(image_path).exists():
        # Test on provided image
        print(f"\nRunning inference on: {image_path}")
        results = model(image_path)
        
        for r in results:
            print(f"\nDetections:")
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]
                print(f"  - {label}: {conf:.2%} confidence")
        
        # Save annotated image
        output_path = Path(image_path).stem + '_detected.jpg'
        results[0].save(output_path)
        print(f"\nAnnotated image saved to: {output_path}")
    else:
        # Test on validation set
        data_yaml = find_data_yaml()
        if data_yaml:
            print("\nNo image provided, running validation...")
            metrics = model.val(data=data_yaml)
            print(f"\nmAP50: {metrics.box.map50:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description='VisionAssist Currency Model Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  First time setup:
    python train_currency_model.py --download --api-key YOUR_ROBOFLOW_KEY
    
  Train the model:
    python train_currency_model.py --train
    
  Resume interrupted training:
    python train_currency_model.py --train --resume
    
  Validate the model:
    python train_currency_model.py --validate
    
  Export for deployment:
    python train_currency_model.py --export
    
  Test on an image:
    python train_currency_model.py --test --image path/to/image.jpg
    
  Do everything:
    python train_currency_model.py --all --api-key YOUR_KEY
        """
    )
    
    parser.add_argument('--download', action='store_true', help='Download dataset from Roboflow')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--validate', action='store_true', help='Validate the trained model')
    parser.add_argument('--export', action='store_true', help='Export model for deployment')
    parser.add_argument('--test', action='store_true', help='Test inference')
    parser.add_argument('--all', action='store_true', help='Download, train, and export')
    parser.add_argument('--api-key', type=str, help='Roboflow API key')
    parser.add_argument('--image', type=str, help='Image path for testing')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    
    args = parser.parse_args()
    
    # Override config with command line args
    if args.api_key:
        CONFIG['ROBOFLOW_API_KEY'] = args.api_key
    if args.epochs:
        CONFIG['EPOCHS'] = args.epochs
    if args.batch_size:
        CONFIG['BATCH_SIZE'] = args.batch_size
    
    # Print banner
    print("\n" + "=" * 60)
    print("  VISIONASSIST CURRENCY MODEL TRAINER")
    print("  CBU Capstone Project - Mohammed")
    print("=" * 60)
    
    # Run requested operations
    if args.all:
        download_dataset()
        train_model()
        export_model()
    else:
        if args.download:
            download_dataset()
        if args.train:
            train_model(resume=args.resume)
        if args.validate:
            validate_model()
        if args.export:
            export_model()
        if args.test:
            test_inference(args.image)
    
    # If no action specified, print help
    if not any([args.download, args.train, args.validate, args.export, args.test, args.all]):
        parser.print_help()


if __name__ == '__main__':
    main()
