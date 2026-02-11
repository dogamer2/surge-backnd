import os
import mimetypes
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

DOWNLOAD_ROUTER = APIRouter(prefix="/download", tags=["download"])

@DOWNLOAD_ROUTER.get("/{filename:path}")
async def download_file(filename: str):
    """
    Serve exported files for download
    
    Args:
        filename: Name of the file to download
        
    Returns:
        FileResponse with the file content
    """
    try:
        # Security: only allow specific filename patterns
        allowed_extensions = {'.pptx', '.pdf'}
        if not any(filename.endswith(ext) for ext in allowed_extensions):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Construct file path (should be in exports directory)
        exports_dir = os.path.join(os.path.dirname(__file__), "..", "app_data", "exports")
        file_path = os.path.join(exports_dir, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine MIME type
        if filename.endswith('.pptx'):
            media_type = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
        elif filename.endswith('.pdf'):
            media_type = 'application/pdf'
        else:
            media_type = 'application/octet-stream'
        
        # Read and return file
        def file_iterator():
            with open(file_path, 'rb') as file:
                yield file.read(8192)  # Read in chunks
        
        response = FileResponse(
            file_iterator(),
            media_type=media_type,
            filename=filename,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")