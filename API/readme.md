# QuPath Web Interface

A web interface for QuPath tissue automation using FastAPI and Paquo.

## Setup

1. Create a new Python environment:

```bash
# Using conda (recommended)
conda create -n qupath python=3.9
conda activate qupath

# Or using venv
python -m venv qupath_env
source qupath_env/bin/activate  # On macOS/Linux
# qupath_env\Scripts\activate   # On Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run setup script:

```bash
python setup.py
```

4. Place your QuPath files:
- Put your classifier in `classifiers/pixel_classifiers/tissues_2.json`
- Put your Groovy script in `scripts/automate_export_newest.groovy`

5. Run the server:

```bash
uvicorn main:app --reload
```

## Project Structure

```
qupath_web/
├── main.py              # Main FastAPI application
├── requirements.txt     # Project dependencies
├── setup.py            # Setup script
├── static/             # Static files
│   └── index.html      # Web interface
├── uploads/            # Temporary upload directory
├── processed_data/     # Output directory
├── classifiers/        # QuPath classifiers
│   └── pixel_classifiers/
│       └── tissues_2.json
└── scripts/            # QuPath scripts
    └── automate_export_newest.groovy
```

## Configuration

1. Make sure QuPath is installed on your system
2. Enter the QuPath path in the web interface
3. Upload .tif images for processing

## Troubleshooting

If you encounter numpy import errors:

```bash
pip uninstall numpy pandas scikit-image paquo
pip install numpy==1.23.5
pip install -r requirements.txt
```

## Common Issues

1. QuPath Path Issues:
   - macOS: `/Applications/QuPath-0.5.1-arm64.app/Contents/MacOS/QuPath-0.5.1-arm64`
   - Windows: `C:\Program Files\QuPath\QuPath.exe`

2. Permission Issues:
   ```bash
   # On macOS/Linux
   chmod +x /path/to/QuPath
   ```

3. Import Errors:
   - Try reinstalling dependencies in the correct order
   - Make sure you're using Python 3.9+

## Notes

- The web interface runs on http://localhost:8000
- Processed files are saved in the `processed_data` directory
- Temporary uploads are stored in the `uploads` directory
