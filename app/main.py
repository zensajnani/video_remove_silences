from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os
from typing import List
import tempfile
from .editor import VideoEditor
from .schemas import VideoEdit, EditResponse

app = FastAPI(title="AI Video Editor API")
editor = VideoEditor()

@app.post("/edit-videos/", response_model=EditResponse)
async def edit_videos(files: List[UploadFile] = File(...)):
    # Create temporary directory for uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        video_files = []
        
        # Save uploaded files
        for file in files:
            temp_file_path = os.path.join(temp_dir, file.filename)
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            video_files.append(temp_file_path)
        
        # Process videos
        result = await editor.process_videos(video_files)
        
        # Move output file to a permanent location
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        final_output = os.path.join(output_dir, "ai_output.mp4")
        shutil.move(result["output_file"], final_output)
        
        return EditResponse(
            edited_script=result["edited_script"],
            output_file=final_output
        )

@app.get("/download/{filename}")
async def download_video(filename: str):
    file_path = os.path.join("outputs", filename)
    return FileResponse(file_path)
