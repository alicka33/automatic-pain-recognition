import os
import uuid
from typing import Tuple
from pathlib import Path
from fastapi import UploadFile


class FileHandlerService:
    """
    Service for handling file uploads and temporary file management.
    """

    def __init__(self, temp_dir: str = "/tmp/pain_detection_uploads"):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    async def save_uploaded_video(self, video: UploadFile) -> Tuple[str, str]:
        """Save uploaded video to temp directory."""
        file_extension = os.path.splitext(video.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(self.temp_dir, unique_filename)

        # Write video file
        with open(file_path, "wb") as buffer:
            buffer.write(await video.read())

        file_size = os.path.getsize(file_path)
        print(f"Saved uploaded video: {video.filename} ({file_size} bytes) -> {unique_filename}")

        return file_path, unique_filename

    def cleanup_file(self, file_path: str) -> bool:
        """Delete temporary file."""
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Cleaned up temporary file: {file_path}")
                return True
            except Exception as e:
                print(f"Warning: Could not remove temp file {file_path}: {e}")
                return False
        return False

    def validate_video_file(self, video: UploadFile) -> Tuple[bool, str]:
        """Validate uploaded file is a video."""
        if not video.filename or not video.content_type:
            return False, "Invalid file upload."

        if not video.content_type.startswith('video/'):
            return False, f"Invalid file type: {video.content_type}. Expected video file."

        return True, ""
