"""
Configuration module for the 6D Detection UI
"""

import os
from pathlib import Path

# Base workspace path
WORKSPACE_ROOT = "/home/yeo/Downloads/SG/src/tomo2-imaging/implementation/6dof_detect"

# BOP related paths
BOP_PARENT_PATH = os.path.join(WORKSPACE_ROOT, "bop", "BlenderProc")
CC_TEXTURES_PATH = os.path.join(BOP_PARENT_PATH, "resources", "cctextures")
OUTPUT_BASE_DIR = os.path.join(BOP_PARENT_PATH, "output")

# Flask configuration
FLASK_ENV = "development"
FLASK_DEBUG = True
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# BlenderProc generation defaults
DEFAULT_NUM_SCENES = 2000
DEFAULT_NUM_IMAGES_PER_SCENE = 25
GENERATION_TIMEOUT = 3600  # seconds (1 hour)

# Dataset configuration
SUPPORTED_DATASETS = {
    'hb': {
        'name': 'Custom Dataset',
        'description': 'User custom dataset',
        'model_type': 'reconstructed'
    },
    'tless': {
        'name': 'T-LESS',
        'description': 'Texture-less objects',
        'model_type': 'cad'
    },
    'ycbv': {
        'name': 'YCB-V',
        'description': 'YCB Video dataset',
        'model_type': 'reconstructed'
    },
    'tyol': {
        'name': 'TYOL',
        'description': 'Custom dataset',
        'model_type': 'reconstructed'
    }
}

# Required files for each dataset
REQUIRED_DATASET_FILES = [
    'camera.json',
    'test_targets_bop19.json'
]

# Flask CORS configuration
CORS_CONFIG = {
    "origins": ["*"],
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type"]
}
