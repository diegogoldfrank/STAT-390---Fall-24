from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import shutil
from typing import List
from paquo.projects import QuPathProject
from paquo.images import QuPathImageType
from paquo.java import start_jvm
import jpype.imports
import jpype
import tempfile
import logging
import traceback

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Start JVM if not already started
if not jpype.isJVMStarted():
    start_jvm()

# Create necessary directories
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed_data"
CLASSIFIER_DIR = BASE_DIR / "classifiers/pixel_classifiers"
SCRIPTS_DIR = BASE_DIR / "scripts"

for dir_path in [STATIC_DIR, UPLOAD_DIR, PROCESSED_DIR, CLASSIFIER_DIR, SCRIPTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def run_groovy_script(qp_project, image_entry, script_path: Path, classifier_path: Path):
    """Run a Groovy script on a QuPath image using paquo"""
    try:
        logger.info(f"Reading Groovy script from: {script_path}")
        with open(script_path, 'r') as f:
            script_content = f.read()

        # Prepare the script with our variables
        modified_script = f"""
            // Set up project variables
            def PROJECT_BASE_DIR = '{str(PROCESSED_DIR)}'
            def outputDir = '{str(PROCESSED_DIR)}'
            def classifierPath = '{str(classifier_path)}'
            
            // Make sure directories exist
            new File(outputDir).mkdirs()
            
            {script_content}
        """

        logger.info("Running script...")
        
        # Get the Java gateway
        gateway = jpype.JImplements('qupath.lib.scripting.ScriptParameters$ScriptParameterValueResolver')()
        
        # Run the script using paquo's built-in method
        image_entry._image.runScript('Groovy', modified_script, gateway)
        
        logger.info("Script execution completed")
        return True

    except Exception as e:
        logger.error(f"Error in run_groovy_script: {str(e)}")
        logger.error(traceback.format_exc())
        return False

async def save_upload(upload_file: UploadFile, destination: Path):
    """Save uploaded file to destination with proper error handling"""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with destination.open("wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)
        finally:
            if upload_file.file:
                upload_file.file.close()
        
        logger.info(f"Successfully saved file to {destination}")
        return destination
        
    except Exception as e:
        logger.error(f"Error saving file {upload_file.filename}: {str(e)}")
        if destination.exists():
            try:
                destination.unlink()
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up partial file: {cleanup_error}")
        raise

app = FastAPI(title="QuPath Tissue Automation")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_files(images: List[UploadFile] = File(...)):
    """Handle file uploads and processing"""
    cleanup_required = []
    try:
        results = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "project"
            project_dir.mkdir()
            
            groovy_script = SCRIPTS_DIR / "automate_export_newest.groovy"
            classifier_path = CLASSIFIER_DIR / "tissues_2.json"
            
            if not groovy_script.exists():
                raise ValueError(f"Groovy script not found at {groovy_script}")
            if not classifier_path.exists():
                raise ValueError(f"Classifier not found at {classifier_path}")
            
            logger.info("Creating QuPath project...")
            with QuPathProject(project_dir / "project.qpproj", mode='x') as qp:
                try:
                    for img in images:
                        try:
                            img_path = UPLOAD_DIR / img.filename
                            cleanup_required.append(img_path)
                            await save_upload(img, img_path)
                            
                            logger.info(f"Processing image: {img_path}")
                            
                            # Add image to project using paquo
                            entry = qp.add_image(img_path)
                            if isinstance(entry, list):
                                entry = entry[0]
                            
                            # Set image type using paquo's method
                            entry.image_type = QuPathImageType.BRIGHTFIELD_H_E
                            
                            # Run the script
                            if run_groovy_script(qp, entry, groovy_script, classifier_path):
                                roi_pattern = f"{img_path.stem}_ROI_*.tif"
                                roi_files = list(PROCESSED_DIR.glob(roi_pattern))
                                
                                if roi_files:
                                    results.append({
                                        "image": img.filename,
                                        "status": "processed",
                                        "output_paths": [f.name for f in roi_files]
                                    })
                                else:
                                    results.append({
                                        "image": img.filename,
                                        "status": "warning",
                                        "message": "No ROIs generated"
                                    })
                            else:
                                results.append({
                                    "image": img.filename,
                                    "status": "error",
                                    "error": "Script execution failed"
                                })
                                
                        except Exception as e:
                            logger.error(f"Error processing {img.filename}: {str(e)}")
                            logger.error(traceback.format_exc())
                            results.append({
                                "image": img.filename,
                                "status": "error",
                                "error": str(e)
                            })
                
                except Exception as e:
                    logger.error(f"Error setting up project: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise

        return {
            "status": "success",
            "message": "Processing complete",
            "results": results
        }

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)
        
    finally:
        for file_path in cleanup_required:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up {file_path}: {cleanup_error}")

@app.get("/download/{filename}")
async def download_processed_file(filename: str):
    try:
        file_path = PROCESSED_DIR / filename
        if not file_path.exists():
            return JSONResponse({
                "status": "error",
                "message": f"File not found: {filename}"
            }, status_code=404)
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return STATIC_DIR.joinpath("index.html").read_text()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)