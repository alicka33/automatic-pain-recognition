import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import config
from services.model_loader import ModelLoader
from services.pain_detection_service import PainDetectionService
from services.file_handler import FileHandlerService


model_loader = ModelLoader()
file_handler = FileHandlerService()
pain_detection_service = None


app = FastAPI(
    title="Pain Detection API",
    description="API for pain level classification based on facial video analysis.",
    version="2.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Initialize services on server startup."""
    global pain_detection_service

    success, error_message = model_loader.load_assets()

    if success:
        pain_detection_service = PainDetectionService(
            model_loader.get_inference_helper()
        )
    else:
        print(f"Failed to initialize services: {error_message}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Pain Detection Server is running",
        "status": "ready" if model_loader.is_loaded() else "error",
        "model_type": config.MODEL_TYPE,
        "num_classes": config.NUM_CLASSES,
        "endpoints": {
            "upload": "/upload-video",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check with model status."""
    is_healthy = model_loader.is_loaded()

    service_info = {}
    if pain_detection_service:
        service_info = pain_detection_service.get_service_info()

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": is_healthy,
        "device": model_loader.get_device_info(),
        "config": service_info
    }


@app.post("/upload-video")
async def upload_video(video: UploadFile = File(...)):
    """Process uploaded video and return pain classification."""
    if not model_loader.is_loaded() or pain_detection_service is None:
        raise HTTPException(
            status_code=503,
            detail="Server unavailable: ML model could not be loaded. Check model files."
        )

    is_valid, error_message = file_handler.validate_video_file(video)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)

    file_path = None
    try:
        file_path, unique_filename = await file_handler.save_uploaded_video(video)

        print(f"Processing video: {video.filename}")

        result = pain_detection_service.analyze_video(file_path)

        print(f"Analysis complete: {result['painLevel']} "
              f"(confidence: {result['confidence']:.3f}, frames: {result['numFrames']})")

        return JSONResponse(content=result)

    except Exception as e:
        print(f"Error during video analysis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: Video analysis failed. Details: {str(e)}"
        )

    finally:
        if file_path:
            file_handler.cleanup_file(file_path)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        log_level="info"
    )
