"""
heyjivu Modal Raw Video Processing Pipeline
App Name: aura-processing-pipeline

Consolidated dispatch: 2 calls per video instead of 17.
  1. analyze_and_extract — ffprobe + audio extract + frame extraction (one R2 download)
  2. render_all — main render + subtitle burn + shorts (one R2 download)
"""

import modal
import subprocess
import tempfile
import json
import os
import time
import fastapi
import shlex

from modal_common import with_retry, send_callback, get_r2_client, download_from_r2, upload_to_r2, detect_encoder

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "fonts-dejavu-core")
    .pip_install("fastapi", "boto3", "botocore", "httpx")
        .add_local_dir("./modal_common", "/root/modal_common")
)

nvidia_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "fonts-dejavu-core")
    .pip_install("fastapi", "boto3", "botocore", "httpx")
        .add_local_dir("./modal_common", "/root/modal_common")
)

app = modal.App("aura-processing-pipeline", image=image)

@app.function()
@modal.fastapi_endpoint(method="GET")
def health():
    return {"status": "ok", "app": "aura-processing-pipeline"}

# ── Consolidated Step 1: Analyze + Extract ────────────────────────────────────

@app.function(timeout=300, memory=4096)
def analyze_and_extract(payload: dict):
    """
    Single R2 download → ffprobe + audio extract + optional frame extraction.
    Returns: VideoInfoJson, HasAudio, AudioKey (if audio), FrameKeys (if requested)
    """
    print(f"DEBUG_LOG [analyze_and_extract] Starting with job_id: {payload.get('job_id')}")
    callback_url = payload["callback_url"]
    callback_secret = payload["callback_secret"]
    r2_config = json.loads(payload.get("r2_config", "{}"))
    client = get_r2_client(r2_config)
    bucket = r2_config.get("bucket_name", "")
    input_key = payload.get("input_key")
    output_folder = payload.get("output_folder", "")
    settings = json.loads(payload.get("settings", "{}"))
    extract_audio_flag = settings.get("extract_audio", True)
    audio_format = settings.get("audio_format", "wav")
    frame_timestamps = settings.get("frame_timestamps", [])

    print(f"DEBUG_LOG [analyze_and_extract] parsed configuration. Input key: {input_key}, bucket: {bucket}")
    try:
        print(f"DEBUG_LOG [analyze_and_extract] Creating temporary directory...")
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp4")
            print(f"DEBUG_LOG [analyze_and_extract] Temp directory created at: {tmpdir}")
            print(f"DEBUG_LOG [analyze_and_extract] Downloading raw video from R2 key: {input_key} -> {input_path}")
            download_from_r2(client, bucket, input_key, input_path)
            print(f"DEBUG_LOG [analyze_and_extract] Download complete. File size: {os.path.getsize(input_path)} bytes")

            outputs = {}

            # ── ffprobe ────────────────────────────────────────────
            print(f"DEBUG_LOG [analyze_and_extract] Running ffprobe on: {input_path}")
            result = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", input_path],
                capture_output=True, text=True, timeout=60, check=True)
            print(f"DEBUG_LOG [analyze_and_extract] ffprobe completed. Exit code: {result.returncode}")
            probe_data = json.loads(result.stdout)
            outputs["VideoInfoJson"] = result.stdout
            duration = float(probe_data.get("format", {}).get("duration", 0))
            print(f"DEBUG_LOG [analyze_and_extract] Probe duration: {duration}s")

            # Audio stream detection
            has_audio = any(s.get("codec_type") == "audio" for s in probe_data.get("streams", []))
            outputs["HasAudio"] = "true" if has_audio else "false"
            print(f"DEBUG_LOG [analyze_and_extract] has_audio: {has_audio}")

            # ── Audio extraction ───────────────────────────────────
            if extract_audio_flag and has_audio:
                audio_path = os.path.join(tmpdir, f"audio.{audio_format}")
                codec = "libmp3lame" if audio_format == "mp3" else "pcm_s16le"
                print(f"DEBUG_LOG [analyze_and_extract] Starting audio extraction to: {audio_path} using codec: {codec}")
                audio_result = subprocess.run(["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", codec, audio_path],
                    capture_output=True, text=True, timeout=280, check=True)
                print(f"DEBUG_LOG [analyze_and_extract] Audio extraction completed. Exit code: {audio_result.returncode}, file size: {os.path.getsize(audio_path)} bytes")
                audio_key = f"{output_folder}audio_{payload['job_id']}.{audio_format}"
                print(f"DEBUG_LOG [analyze_and_extract] Uploading audio to R2: {audio_key}...")
                upload_to_r2(client, bucket, audio_key, audio_path, f"audio/{audio_format}")
                outputs["AudioKey"] = audio_key
                print(f"DEBUG_LOG [analyze_and_extract] Audio upload complete.")

            # ── Frame extraction for vision transcription ──────────
            if frame_timestamps:
                print(f"DEBUG_LOG [analyze_and_extract] Extracting {len(frame_timestamps)} frames at timestamps: {frame_timestamps}")
                frame_keys = []
                for i, ts in enumerate(frame_timestamps):
                    frame_path = os.path.join(tmpdir, f"frame_{i}.jpg")
                    print(f"DEBUG_LOG [analyze_and_extract] Extracting frame {i} at {ts}s to {frame_path}")
                    frame_result = subprocess.run(["ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", input_path,
                         "-vframes", "1", "-q:v", "2", frame_path],
                        capture_output=True, text=True, timeout=60, check=True)
                    print(f"DEBUG_LOG [analyze_and_extract] Frame {i} extract exit code: {frame_result.returncode}, size: {os.path.getsize(frame_path)} bytes")
                    key = f"{output_folder}frame_{i}_{payload['job_id']}.jpg"
                    print(f"DEBUG_LOG [analyze_and_extract] Uploading frame {i} to: {key}...")
                    upload_to_r2(client, bucket, key, frame_path, "image/jpeg")
                    frame_keys.append(key)
                outputs["FrameKeys"] = json.dumps(frame_keys)
                print(f"DEBUG_LOG [analyze_and_extract] Frame extraction complete.")

        print(f"DEBUG_LOG [analyze_and_extract] Sending success callback to: {callback_url}")
        send_callback(callback_url, callback_secret, {
            "success": True,
            "outputFilesJson": json.dumps(outputs)
        })
        print(f"DEBUG_LOG [analyze_and_extract] Success callback sent.")
    except Exception as e:
        import traceback
        print("DEBUG_LOG [analyze_and_extract] EXCEPTION OCCURRED:")
        traceback.print_exc()
        print(f"DEBUG_LOG [analyze_and_extract] Sending failure callback with error: {str(e)}")
        send_callback(callback_url, callback_secret, {
            "success": False,
            "errorMessage": str(e)
        })
        print("DEBUG_LOG [analyze_and_extract] Failure callback sent.")

# ── Consolidated Step 2: Render All (Main + Subs + Shorts) ───────────────────

@app.function(image=nvidia_image, gpu="A10G", timeout=720, memory=8192)
def render_all(payload: dict):
    """
    Single R2 download → main render + subtitle burn + shorts creation.
    Returns: MainVideoKey, MainDuration, ShortKeys
    """
    print(f"DEBUG_LOG [render_all] Starting with job_id: {payload.get('job_id')}")
    callback_url = payload["callback_url"]
    callback_secret = payload["callback_secret"]
    r2_config = json.loads(payload.get("r2_config", "{}"))
    client = get_r2_client(r2_config)
    bucket = r2_config.get("bucket_name", "")
    input_key = payload.get("input_key")
    output_folder = payload.get("output_folder", "")
    settings = json.loads(payload.get("settings", "{}"))
    main_args = settings.get("main_args", "")
    short_segments = settings.get("short_segments", [])
    srt_content = settings.get("srt_content", None)
    encoder = detect_encoder()

    print(f"DEBUG_LOG [render_all] Parsed configuration. Input key: {input_key}, bucket: {bucket}, detected encoder: {encoder}")
    try:
        print(f"DEBUG_LOG [render_all] Creating temporary directory...")
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp4")
            print(f"DEBUG_LOG [render_all] Temp directory created at: {tmpdir}")
            print(f"DEBUG_LOG [render_all] Downloading raw video: {input_key} -> {input_path}")
            download_from_r2(client, bucket, input_key, input_path)
            print(f"DEBUG_LOG [render_all] Download complete. File size: {os.path.getsize(input_path)} bytes")
            
            # Normalize paths to use forward slashes (avoids backslash escaping issues with shlex.split on Windows)
            input_path = input_path.replace("\\", "/")
            print(f"DEBUG_LOG [render_all] Normalized input_path: {input_path}")

            outputs = {}
            job_id = payload["job_id"]

            # ── 1. Main video processing ──────────────────────────
            main_output_path = os.path.join(tmpdir, "main_processed.mp4").replace("\\", "/")
            print(f"DEBUG_LOG [render_all] Main output path: {main_output_path}")
            if main_args:
                ff_args = main_args.replace("{input}", input_path)
                ff_args = ff_args.replace("{encoder}", encoder)
                cmd = ["ffmpeg", "-y"] + shlex.split(ff_args) + [main_output_path]
                print(f"DEBUG_LOG [render_all] Running main render command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=540, check=True)
                print(f"DEBUG_LOG [render_all] Main render completed. Exit code: {result.returncode}")
                if result.returncode != 0:
                    print(f"DEBUG_LOG [render_all] Main render FAILED! stderr: {result.stderr}")
                    raise RuntimeError(f"Main render failed:\n{result.stderr[-3000:]}")
                else:
                    print(f"DEBUG_LOG [render_all] Main render file size: {os.path.getsize(main_output_path)} bytes")
            else:
                print(f"DEBUG_LOG [render_all] No main_args provided. Skipping main render, main_output_path = input_path")
                main_output_path = input_path

            # ── 2. Optional subtitle burn ─────────────────────────
            if srt_content and srt_content.strip():
                srt_path = os.path.join(tmpdir, "subtitles.srt").replace("\\", "/")
                print(f"DEBUG_LOG [render_all] Writing subtitles.srt content (length: {len(srt_content)}) to: {srt_path}")
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                subs_output_path = os.path.join(tmpdir, "main_subtitled.mp4").replace("\\", "/")
                codec, extra = ("h264_nvenc", "-preset p4 -rc vbr -cq 23") if encoder == "h264_nvenc" else ("libx264", "-preset fast -crf 23")
                # Burn subtitles using relative filename and setting cwd=tmpdir to avoid Windows path issues with FFmpeg filtergraph
                cmd_subs = ["ffmpeg", "-y", "-i", main_output_path, "-vf", "subtitles=subtitles.srt",
                     "-c:v", codec] + extra.split() + ["-c:a", "aac", "-b:a", "192k", subs_output_path]
                print(f"DEBUG_LOG [render_all] Running subtitles burn command (in cwd={tmpdir}): {' '.join(cmd_subs)}")
                subs_result = subprocess.run(cmd_subs, capture_output=True, text=True, timeout=540, check=True, cwd=tmpdir)
                print(f"DEBUG_LOG [render_all] Subtitle burn completed. Exit code: {subs_result.returncode}")
                if os.path.exists(subs_output_path) and os.path.getsize(subs_output_path) > 1024:
                    print(f"DEBUG_LOG [render_all] Subtitle burned file exists. Size: {os.path.getsize(subs_output_path)} bytes. Updating main_output_path.")
                    main_output_path = subs_output_path
                else:
                    print(f"DEBUG_LOG [render_all] Subtitle burned file is missing or too small.")

            # Upload main video
            main_key = settings.get("main_output_key") or f"{output_folder}main_{job_id}.mp4"
            print(f"DEBUG_LOG [render_all] Uploading main video to R2 key: {main_key.encode('ascii', 'replace').decode('ascii')} from: {main_output_path}")
            upload_to_r2(client, bucket, main_key, main_output_path, "video/mp4")
            outputs["MainVideoKey"] = main_key
            print(f"DEBUG_LOG [render_all] Main video upload complete.")

            # Upload main script file
            script_content = settings.get("script_content")
            if script_content:
                main_folder = "/".join(main_key.split("/")[:-1])
                script_path = os.path.join(tmpdir, "script.txt")
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(script_content)
                script_key = f"{main_folder}/script.txt"
                print(f"DEBUG_LOG [render_all] Uploading main script to R2 key: {script_key.encode('ascii', 'replace').decode('ascii')}")
                upload_to_r2(client, bucket, script_key, script_path, "text/plain")

            # Verify duration
            print(f"DEBUG_LOG [render_all] Running ffprobe to verify main video duration...")
            dur_result = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", main_output_path],
                capture_output=True, text=True, timeout=30, check=True)
            if dur_result.returncode == 0:
                info = json.loads(dur_result.stdout)
                outputs["MainDuration"] = str(info.get("format", {}).get("duration", "0"))
                print(f"DEBUG_LOG [render_all] Verified duration: {outputs['MainDuration']}s")
            else:
                outputs["MainDuration"] = "0"
                print(f"DEBUG_LOG [render_all] ffprobe failed to get duration, default to 0")

            # ── 3. Shorts generation ──────────────────────────────
            print(f"DEBUG_LOG [render_all] Processing {len(short_segments)} short clips...")
            short_keys = []
            sw, sh = 1080, 1920
            for idx, seg in enumerate(short_segments):
                start = float(seg.get("start", 0))
                end = float(seg.get("end", 15))
                dur = end - start
                short_path = os.path.join(tmpdir, f"short_{idx+1}.mp4").replace("\\", "/")
                short_filter = (
                    f"scale={sw}:{sh}:force_original_aspect_ratio=decrease,"
                    f"pad={sw}:{sh}:(ow-iw)/2:(oh-ih)/2:color=black,"
                    "setsar=1,format=yuv420p"
                )
                cmd = [
                    "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                    "-i", input_path,
                    "-vf", short_filter,
                    "-c:v", encoder, "-c:a", "aac", short_path
                ]
                print(f"DEBUG_LOG [render_all] Creating short clip {idx+1} ({start}s to {end}s, dur={dur}s). Command: {' '.join(cmd)}")
                short_result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if short_result.returncode != 0 and encoder == "h264_nvenc":
                    print(f"DEBUG_LOG [render_all] Clip {idx+1} GPU render failed. stderr: {short_result.stderr[-3000:]}")
                    cmd = [
                        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                        "-i", input_path,
                        "-vf", short_filter,
                        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        "-c:a", "aac", short_path
                    ]
                    print(f"DEBUG_LOG [render_all] Retrying short clip {idx+1} with CPU encoder. Command: {' '.join(cmd)}")
                    short_result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                if short_result.returncode != 0:
                    print(f"DEBUG_LOG [render_all] Clip {idx+1} render FAILED! stderr: {short_result.stderr[-3000:]}")
                    raise RuntimeError(f"Short clip {idx+1} render failed:\n{short_result.stderr[-3000:]}")
                print(f"DEBUG_LOG [render_all] Clip {idx+1} creation exit code: {short_result.returncode}")

                if os.path.exists(short_path) and os.path.getsize(short_path) > 1024:
                    key = seg.get("output_key")
                    if not key:
                        shorts_folder = settings.get("shorts_output_folder") or output_folder
                        key = f"{shorts_folder}short_{idx+1}_{job_id}.mp4"
                    print(f"DEBUG_LOG [render_all] Uploading clip {idx+1} to key: {key.encode('ascii', 'replace').decode('ascii')} (size: {os.path.getsize(short_path)} bytes)")
                    upload_to_r2(client, bucket, key, short_path, "video/mp4")
                    short_keys.append(key)
                else:
                    print(f"DEBUG_LOG [render_all] WARNING: Clip {idx+1} file not found or too small!")

            outputs["ShortKeys"] = json.dumps(short_keys)
            print(f"DEBUG_LOG [render_all] All short clips processed. count: {len(short_keys)}")

        print(f"DEBUG_LOG [render_all] Sending success callback to: {callback_url}")
        send_callback(callback_url, callback_secret, {
            "success": True,
            "outputFilesJson": json.dumps(outputs)
        })
        print(f"DEBUG_LOG [render_all] Success callback sent.")
    except Exception as e:
        import traceback
        print("DEBUG_LOG [render_all] EXCEPTION OCCURRED:")
        traceback.print_exc()
        print(f"DEBUG_LOG [render_all] Sending failure callback to: {callback_url} with error: {str(e)}")
        send_callback(callback_url, callback_secret, {
            "success": False,
            "errorMessage": str(e)
        })
        print("DEBUG_LOG [render_all] Failure callback sent.")

# ── (Keep for backward compatibility) ─────────────────────────────────────────

@app.function(timeout=720, memory=4096)
def extract_thumbnail(payload: dict):
    """Legacy: single frame extraction (no GPU needed)."""
    print(f"DEBUG_LOG [extract_thumbnail] Starting with job_id: {payload.get('job_id')}")
    callback_url = payload["callback_url"]
    callback_secret = payload["callback_secret"]
    r2_config = json.loads(payload.get("r2_config", "{}"))
    client = get_r2_client(r2_config)
    bucket = r2_config.get("bucket_name", "")
    input_key = payload.get("input_key")
    output_folder = payload.get("output_folder", "")
    settings = json.loads(payload.get("settings", "{}"))
    timestamp_secs = float(settings.get("timestamp_secs", 1))
    output_format = settings.get("output_format", "jpg")

    print(f"DEBUG_LOG [extract_thumbnail] Parsed configuration. Input key: {input_key}, timestamp: {timestamp_secs}s, format: {output_format}")
    try:
        print(f"DEBUG_LOG [extract_thumbnail] Creating temporary directory...")
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, f"thumb.{output_format}").replace("\\", "/")
            print(f"DEBUG_LOG [extract_thumbnail] Temp directory created at: {tmpdir}")
            print(f"DEBUG_LOG [extract_thumbnail] Downloading raw video: {input_key} -> {input_path}")
            download_from_r2(client, bucket, input_key, input_path)
            input_path = input_path.replace("\\", "/")
            print(f"DEBUG_LOG [extract_thumbnail] Download complete. File size: {os.path.getsize(input_path)} bytes")

            cmd = ["ffmpeg", "-y", "-ss", f"{timestamp_secs:.3f}", "-i", input_path,
                 "-vframes", "1", "-q:v", "2", "-vsync", "vfr", output_path]
            print(f"DEBUG_LOG [extract_thumbnail] Running thumbnail extract command: {' '.join(cmd)}")
            thumb_result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
            print(f"DEBUG_LOG [extract_thumbnail] Thumbnail extract completed. Exit code: {thumb_result.returncode}")

            output_key = f"{output_folder}thumb_{payload['job_id']}.{output_format}"
            print(f"DEBUG_LOG [extract_thumbnail] Uploading thumbnail to R2 key: {output_key.encode('ascii', 'replace').decode('ascii')} from: {output_path}")
            upload_to_r2(client, bucket, output_key, output_path, f"image/{output_format}")
            print(f"DEBUG_LOG [extract_thumbnail] Thumbnail upload complete.")

        print(f"DEBUG_LOG [extract_thumbnail] Sending success callback to: {callback_url}")
        send_callback(callback_url, callback_secret, {
            "success": True,
            "outputFilesJson": json.dumps({"ThumbnailKey": output_key})
        })
        print(f"DEBUG_LOG [extract_thumbnail] Success callback sent.")
    except Exception as e:
        import traceback
        print("DEBUG_LOG [extract_thumbnail] EXCEPTION OCCURRED:")
        traceback.print_exc()
        print(f"DEBUG_LOG [extract_thumbnail] Sending failure callback to: {callback_url} with error: {str(e)}")
        send_callback(callback_url, callback_secret, {
            "success": False, "errorMessage": str(e)
        })
        print("DEBUG_LOG [extract_thumbnail] Failure callback sent.")

# ── Dispatch Endpoint ────────────────────────────────────────────────────────

@app.function(secrets=[modal.Secret.from_name("aura-secrets")])
@modal.fastapi_endpoint(method="POST")
def dispatch(body: dict, request: fastapi.Request):
    expected_secret = os.environ.get("MODAL_API_SECRET")
    secret = request.headers.get("X-Modal-Secret")
    if not expected_secret or secret != expected_secret:
        raise fastapi.HTTPException(status_code=401, detail="Unauthorized")

    step_name = body.get("step_name")
    function_map = {
        "AnalyzeAndExtract": analyze_and_extract,
        "RenderAll":         render_all,
        "ExtractAudio":      analyze_and_extract,  # legacy → consolidated
        "ProcessVideo":      render_all,            # legacy → consolidated
        "OverlaySubtitles":  render_all,            # legacy → consolidated
        "GenerateShorts":    render_all,            # legacy → consolidated
        "CreateShort":       render_all,            # legacy → consolidated
        "ExtractThumbnail":  extract_thumbnail,
        "GetVideoInfo":      analyze_and_extract,  # legacy → consolidated
        "ProcessTo1080p":    render_all,            # legacy → consolidated
    }
    fn = function_map.get(step_name)
    if fn is None:
        raise fastapi.HTTPException(status_code=400, detail=f"Unknown step: {step_name}")
    fn.spawn(body)
    return {"dispatched": True}

