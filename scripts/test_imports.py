#!/usr/bin/env python3
"""
Simple test script to verify imports are working.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

print("🧪 Testing imports...")

try:
    from config import get_config
    print("✅ config module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import config: {e}")
    exit(1)

try:
    from data import FMRIDataProcessor, SMRIDataProcessor
    print("✅ data module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import data: {e}")
    exit(1)

try:
    from models import SingleAtlasTransformer, SMRITransformer, CrossAttentionTransformer
    print("✅ models module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import models: {e}")
    exit(1)

try:
    from training import Trainer, EarlyStopping, create_data_loaders
    print("✅ training module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import training: {e}")
    exit(1)

try:
    from evaluation import evaluate_model, create_cv_visualizations
    print("✅ evaluation module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import evaluation: {e}")
    exit(1)

try:
    from utils import get_device, run_cross_validation
    print("✅ utils module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import utils: {e}")
    exit(1)

print("\n🎉 All imports successful!")
print("✅ The repository is properly set up and ready to use!")
print("\nFor Google Colab usage:")
print("- Make sure to mount your Google Drive")
print("- Use the paths specified in COLAB_GUIDE.md")
print("- All scripts should work with python3 scripts/script_name.py") 