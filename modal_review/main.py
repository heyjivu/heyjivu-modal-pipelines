"""
AuraDesk Modal Review Pipeline
App Name: aura-review-pipeline

Async dispatch: handles trimming, thumbnail generation, Gemini description,
and folder movement on Google Drive (egress-free), then POSTs callback.
"""

import modal
import subprocess
import tempfile
import json
import os
import time
import fastapi
import httpx
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

from modal_common import with_retry

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("fastapi", "httpx", "google-api-python-client", "google-auth", "google-auth-oauthlib", "google-auth-httplib2")
    .copy_local_dir("./modal_common", "/root/modal_common")
)

app = modal.App("aura-review-pipeline", image=image)

# ── Google Drive Helpers ──────────────────────────────────────────────────────

def get_drive_service(client_id: str, client_secret: str, refresh_token: str):
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret
    )
    return build("drive", "v3", credentials=creds)

def download_file(service, file_id: str, dest_path: str):
    request = service.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

def upload_file(service, src_path: str, filename: str, parent_folder_id: str, mime_type: str = "application/octet-stream") -> str:
    file_metadata = {
        "name": filename,
        "parents": [parent_folder_id]
    }
    media = MediaFileUpload(src_path, mimetype=mime_type, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    return file.get("id")

def delete_file(service, file_id: str):
    try:
        service.files().delete(fileId=file_id).execute()
    except Exception as e:
        print(f"WARNING: Failed to delete old file {file_id}: {e}")

def move_file_or_folder(service, file_id: str, new_parent_id: str):
    file = service.files().get(fileId=file_id, fields="parents").execute()
    previous_parents = ",".join(file.get("parents", []))
    file = service.files().update(
        fileId=file_id,
        addParents=new_parent_id,
        removeParents=previous_parents,
        fields="id, parents"
    ).execute()
    return file

# ── Gemini Helper ─────────────────────────────────────────────────────────────

def generate_gemini_description(prompt: str, api_key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
    body = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1000
        }
    }
    headers = {"Content-Type": "application/json"}
    response = httpx.post(url, json=body, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        raise ValueError(f"Gemini API returned invalid response: {data}")

# ── Callback Helper ───────────────────────────────────────────────────────────

def send_callback(callback_url: str, payload: dict, callback_secret: str = None):
    if not callback_url:
        return
    headers = {}
    if callback_secret:
        headers["X-Modal-Secret"] = callback_secret
    print(f"Sending callback to {callback_url} with payload keys: {list(payload.keys())} and headers: {list(headers.keys())}")
    with_retry(lambda: httpx.post(callback_url, json=payload, headers=headers, timeout=30))

# ── Orchestrated Worker Function ──────────────────────────────────────────────

@app.function(timeout=600, memory=4096)
def process_review_job(payload: dict):
    video_id = payload.get("video_id")
    is_finalize = payload.get("is_finalize", False)
    generate_video = payload.get("generate_video", False)
    generate_thumbnail = payload.get("generate_thumbnail", False)
    generate_description = payload.get("generate_description", False)
    start_time = payload.get("start_time")
    end_time = payload.get("end_time")
    thumbnail_timestamp = payload.get("thumbnail_timestamp")
    title = payload.get("title")
    description = payload.get("description")
    folder_id = payload.get("folder_id")
    video_file_id = payload.get("video_file_id")
    thumbnail_file_id = payload.get("thumbnail_file_id")
    target_folder_id = payload.get("target_folder_id")
    
    oauth = payload.get("oauth_credentials", {})
    client_id = oauth.get("client_id")
    client_secret = oauth.get("client_secret")
    refresh_token = oauth.get("refresh_token")
    
    gemini_api_key = payload.get("gemini_api_key")
    callback_url = payload.get("callback_url")
    callback_secret = payload.get("callback_secret")

    print(f"DEBUG_LOG [process_review_job] Started for video_id: {video_id}. is_finalize: {is_finalize}, generate_video: {generate_video}, generate_thumbnail: {generate_thumbnail}, generate_description: {generate_description}")
    try:
        if not client_id or not client_secret or not refresh_token:
            print("DEBUG_LOG [process_review_job] ERROR: Missing Google Drive OAuth credentials.")
            raise ValueError("Missing Google Drive OAuth credentials.")

        # Build Google Drive Service
        print("DEBUG_LOG [process_review_job] Building Google Drive service...")
        service = get_drive_service(client_id, client_secret, refresh_token)
        print("DEBUG_LOG [process_review_job] Google Drive service built successfully.")

        # ── Case A: Finalize & Move Folder ────────────────────────────────────
        if is_finalize:
            if not folder_id:
                raise ValueError("folder_id is required to finalize.")
            if not target_folder_id:
                raise ValueError("target_folder_id is required to finalize.")

            print(f"DEBUG_LOG [process_review_job] Finalizing review. Moving folder {folder_id} to parent {target_folder_id}")
            move_file_or_folder(service, folder_id, target_folder_id)
            print(f"DEBUG_LOG [process_review_job] Folder moved successfully.")

            print(f"DEBUG_LOG [process_review_job] Sending finalize success callback to: {callback_url}")
            send_callback(callback_url, {
                "videoId": video_id,
                "isFinalize": True,
                "success": True
            }, callback_secret)
            print(f"DEBUG_LOG [process_review_job] Finalize success callback sent.")
            return

        # ── Case B: Regenerate video parts, thumbnail or description ──────────
        updated_video_file_id = video_file_id
        updated_thumbnail_file_id = thumbnail_file_id
        updated_description = description

        # 1. Handle Video Trimming / Thumbnail Extraction (needs file download)
        need_download = (generate_video and start_time is not None and end_time is not None) or generate_thumbnail
        print(f"DEBUG_LOG [process_review_job] Need file download: {need_download}")
        
        if need_download:
            if not video_file_id:
                raise ValueError("video_file_id is required for video or thumbnail generation.")

            print("DEBUG_LOG [process_review_job] Creating temporary directory...")
            with tempfile.TemporaryDirectory() as tmpdir:
                input_video_path = os.path.join(tmpdir, "input.mp4")
                print(f"DEBUG_LOG [process_review_job] Temp directory: {tmpdir}")
                print(f"DEBUG_LOG [process_review_job] Downloading original video {video_file_id} to {input_video_path}...")
                download_file(service, video_file_id, input_video_path)
                print(f"DEBUG_LOG [process_review_job] Download complete. Size: {os.path.getsize(input_video_path)} bytes")

                source_video_path = input_video_path
                new_video_created = False

                # Handle Trimming
                if generate_video and start_time is not None and end_time is not None:
                    duration = float(end_time) - float(start_time)
                    trimmed_video_path = os.path.join(tmpdir, "trimmed.mp4")
                    print(f"DEBUG_LOG [process_review_job] Trimming video from {start_time}s with duration {duration}s -> {trimmed_video_path}...")
                    
                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", str(start_time),
                        "-t", str(duration),
                        "-i", input_video_path,
                        "-c:v", "libx264",
                        "-c:a", "aac",
                        "-preset", "fast",
                        "-crf", "23",
                        trimmed_video_path
                    ]
                    print(f"DEBUG_LOG [process_review_job] Command: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
                    print(f"DEBUG_LOG [process_review_job] Trim completed. Exit code: {result.returncode}")
                    if result.returncode != 0:
                        print(f"DEBUG_LOG [process_review_job] Trim FAILED! stderr: {result.stderr}")
                        raise RuntimeError(f"FFmpeg trim failed:\n{result.stderr}")
                    else:
                        print(f"DEBUG_LOG [process_review_job] Trimmed file size: {os.path.getsize(trimmed_video_path)} bytes")
                    
                    source_video_path = trimmed_video_path
                    new_video_created = True

                # Handle Thumbnail Extraction
                if generate_thumbnail:
                    t_sec = float(thumbnail_timestamp) if thumbnail_timestamp is not None else 2.0
                    thumb_path = os.path.join(tmpdir, "thumbnail.png")
                    print(f"DEBUG_LOG [process_review_job] Extracting thumbnail frame at {t_sec} seconds to {thumb_path}...")
                    
                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", str(t_sec),
                        "-i", source_video_path,
                        "-vframes", "1",
                        "-q:v", "2",
                        thumb_path
                    ]
                    print(f"DEBUG_LOG [process_review_job] Command: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
                    print(f"DEBUG_LOG [process_review_job] Thumbnail extraction completed. Exit code: {result.returncode}")
                    if result.returncode != 0:
                        print(f"DEBUG_LOG [process_review_job] Thumbnail FAILED! stderr: {result.stderr}")
                        raise RuntimeError(f"FFmpeg thumbnail extraction failed:\n{result.stderr}")
                    else:
                        print(f"DEBUG_LOG [process_review_job] Thumbnail file size: {os.path.getsize(thumb_path)} bytes")

                    # Upload new thumbnail
                    print(f"DEBUG_LOG [process_review_job] Uploading new thumbnail to folder {folder_id}...")
                    new_thumb_id = upload_file(service, thumb_path, "thumbnail.png", folder_id, "image/png")
                    print(f"DEBUG_LOG [process_review_job] Thumbnail uploaded. New ID: {new_thumb_id}")
                    
                    # Delete old thumbnail
                    if thumbnail_file_id and thumbnail_file_id != new_thumb_id:
                        print(f"DEBUG_LOG [process_review_job] Deleting old thumbnail {thumbnail_file_id}...")
                        delete_file(service, thumbnail_file_id)
                        print("DEBUG_LOG [process_review_job] Old thumbnail deleted.")

                    updated_thumbnail_file_id = new_thumb_id

                # Upload trimmed video if it was created
                if new_video_created:
                    print(f"DEBUG_LOG [process_review_job] Uploading trimmed video to folder {folder_id}...")
                    new_vid_id = upload_file(service, source_video_path, "video_trimmed.mp4", folder_id, "video/mp4")
                    print(f"DEBUG_LOG [process_review_job] Trimmed video uploaded. New ID: {new_vid_id}")
                    
                    # Delete old video
                    if video_file_id and video_file_id != new_vid_id:
                        print(f"DEBUG_LOG [process_review_job] Deleting old video {video_file_id}...")
                        delete_file(service, video_file_id)
                        print("DEBUG_LOG [process_review_job] Old video deleted.")

                    updated_video_file_id = new_vid_id

        # 2. Handle Description Generation
        if generate_description:
            if not gemini_api_key:
                raise ValueError("Gemini API key is required to generate description.")
            
            prompt = (
                f"Generate a descriptive social media caption and video description for a video titled: '{title or ''}'. "
                "Keep it engaging and concise, and return ONLY the description without any metadata or title."
            )
            print("DEBUG_LOG [process_review_job] Generating description using Gemini Flash...")
            updated_description = generate_gemini_description(prompt, gemini_api_key)
            print(f"DEBUG_LOG [process_review_job] Description generated: {updated_description[:100]}...")

        # ── Callback ──────────────────────────────────────────────────────────
        print(f"DEBUG_LOG [process_review_job] Sending success callback to: {callback_url}")
        send_callback(callback_url, {
            "videoId": video_id,
            "isFinalize": False,
            "success": True,
            "videoFileId": updated_video_file_id,
            "thumbnailFileId": updated_thumbnail_file_id,
            "suggestedDescription": updated_description
        }, callback_secret)
        print("DEBUG_LOG [process_review_job] Success callback sent.")

    except Exception as e:
        import traceback
        print("DEBUG_LOG [process_review_job] EXCEPTION OCCURRED:")
        traceback.print_exc()
        print(f"DEBUG_LOG [process_review_job] Sending failure callback to: {callback_url} with error: {str(e)}")
        send_callback(callback_url, {
            "videoId": video_id,
            "isFinalize": is_finalize,
            "success": False,
            "errorMessage": str(e)
        }, callback_secret)
        print("DEBUG_LOG [process_review_job] Failure callback sent.")

# ── FastAPI Entrypoint ────────────────────────────────────────────────────────

@app.function(secrets=[modal.Secret.from_name("aura-secrets")])
@modal.fastapi_endpoint(method="POST")
def dispatch(body: dict, request: fastapi.Request):
    expected_secret = os.environ.get("MODAL_API_SECRET")
    secret = request.headers.get("X-Modal-Secret")
    if not expected_secret or secret != expected_secret:
        raise fastapi.HTTPException(status_code=401, detail="Unauthorized")

    process_review_job.spawn(body)
    return {"dispatched": True}
