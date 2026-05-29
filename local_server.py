import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["PATH"] = r"C:\Users\Malik Umer\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin" + os.pathsep + os.environ.get("PATH", "")

import asyncio
from fastapi import FastAPI, BackgroundTasks, Request, HTTPException

# Import the modal logic directly from your existing apps
from modal_processing.main import analyze_and_extract, render_all, extract_thumbnail
from modal_review.main import process_review_job
from modal_smart_video.main import assemble_video, generate_thumbnails, generate_shorts
from modal_social_post.main import mix_audio_video

# MODAL_API_SECRET must be set in environment for production
# Local dev can still pass "dev-secret" header for testing

app = FastAPI(title="heyjivu Local Processing Server")

# ── Concurrency Limit ──────────────────────────────────────────────────────────
gpu_semaphore = asyncio.Semaphore(2)

def check_auth(request: Request):
    secret = request.headers.get("X-Modal-Secret") or request.headers.get("Authorization")
    print(f"[AUTH DEBUG] Received secret/header: {secret}")
    if secret and secret.startswith("Bearer "):
        secret = secret[7:]
        print(f"[AUTH DEBUG] Stripped bearer token: {secret}")
    expected = os.environ.get("MODAL_API_SECRET", "sk-aura-abc123xyz")
    print(f"[AUTH DEBUG] Expected MODAL_API_SECRET: {expected}")
    if secret != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

def rewrite_callback_url(body: dict):
    if "callback_url" in body and isinstance(body["callback_url"], str):
        original = body["callback_url"]
        if "localhost" not in original and "127.0.0.1" not in original:
            parts = original.split("/", 3)
            if len(parts) >= 4:
                body["callback_url"] = "http://localhost:5000/" + parts[3]
            print(f"[CALLBACK REWRITE] Rewrote callback URL: {original} -> {body['callback_url']}")

@app.post("/processing/dispatch")
async def processing_dispatch(body: dict, request: Request, background_tasks: BackgroundTasks):
    check_auth(request)
    rewrite_callback_url(body)

    step_name = body.get("step_name")
    
    async def run_task():
        async with gpu_semaphore:
            if step_name in ["AnalyzeAndExtract", "ExtractAudio", "GetVideoInfo"]:
                await asyncio.to_thread(analyze_and_extract.local, body)
            elif step_name in ["RenderAll", "ProcessVideo", "OverlaySubtitles", "GenerateShorts", "CreateShort", "ProcessTo1080p"]:
                await asyncio.to_thread(render_all.local, body)
            elif step_name == "ExtractThumbnail":
                await asyncio.to_thread(extract_thumbnail.local, body)
            else:
                print(f"Unknown step: {step_name}")

    background_tasks.add_task(run_task)
    return {"dispatched": True}

@app.post("/review/dispatch")
async def review_dispatch(body: dict, request: Request, background_tasks: BackgroundTasks):
    check_auth(request)
    rewrite_callback_url(body)

    async def run_task():
        async with gpu_semaphore:
            await asyncio.to_thread(process_review_job.local, body)
            
    background_tasks.add_task(run_task)
    return {"dispatched": True}

@app.post("/smart_video/dispatch")
async def smart_video_dispatch(body: dict, request: Request, background_tasks: BackgroundTasks):
    check_auth(request)
    rewrite_callback_url(body)

    step_name = body.get("step_name")
    
    async def run_task():
        async with gpu_semaphore:
            if step_name == "AssemblingVideo":
                await asyncio.to_thread(assemble_video.local, body)
            elif step_name == "GeneratingThumbnails":
                await asyncio.to_thread(generate_thumbnails.local, body)
            elif step_name == "GeneratingShorts":
                await asyncio.to_thread(generate_shorts.local, body)
            else:
                print(f"Unknown step: {step_name}")
            
    background_tasks.add_task(run_task)
    return {"dispatched": True}

@app.post("/social_post/dispatch")
async def social_post_dispatch(body: dict, request: Request, background_tasks: BackgroundTasks):
    check_auth(request)
    rewrite_callback_url(body)

    step_name = body.get("step_name")
    
    async def run_task():
        async with gpu_semaphore:
            if step_name == "MixAudioVideo":
                await asyncio.to_thread(mix_audio_video.local, body)
            else:
                print(f"Unknown step: {step_name}")
            
    background_tasks.add_task(run_task)
    return {"dispatched": True}

if __name__ == "__main__":
    import uvicorn
    print("Starting heyjivu Local GPU Server on http://localhost:8000")
    uvicorn.run("local_server:app", host="0.0.0.0", port=8000, reload=False)
