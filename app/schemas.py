from pydantic import BaseModel
from typing import List

class VideoEdit(BaseModel):
    video_files: List[str]

class EditResponse(BaseModel):
    original_transcripts: List[str]
    edited_script: str
    output_file: str
