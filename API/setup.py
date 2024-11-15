import os
from pathlib import Path

# Project structure
directories = [
    'static',
    'uploads',
    'processed_data',
    'classifiers/pixel_classifiers',
    'scripts'
]

# Create directories
for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Create empty __init__.py files
for directory in directories:
    init_file = Path(directory) / '__init__.py'
    init_file.touch(exist_ok=True)
