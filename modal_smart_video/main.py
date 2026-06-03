"""
heyjivu Modal Smart Video & AI Video Assembly Pipeline
App Name: aura-smart-video-pipeline

Async dispatch: receives a payload, processes via FFmpeg on GPU,
uploads result to R2, then POSTs callback to the API.
"""

import modal
import subprocess
import tempfile
import json
import os
import time
import fastapi
import base64
import httpx

from modal_common import with_retry, send_callback, get_r2_client, download_from_r2, upload_to_r2, detect_encoder

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "fonts-dejavu-core")
    .pip_install("boto3", "botocore", "httpx", "fastapi[standard]")
        .add_local_dir("./modal_common", "/root/modal_common")
)

nvidia_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "fonts-dejavu-core")
    .pip_install("boto3", "botocore", "httpx", "fastapi[standard]")
        .add_local_dir("./modal_common", "/root/modal_common")
)

app = modal.App("aura-smart-video-pipeline", image=image)

def _matching_scene_image(images, scene_idx, zero_based_idx):
    fallback_candidates = {zero_based_idx, zero_based_idx + 1}
    fallback = None
    for image in images:
        try:
            value = int(image.get("sceneIndex"))
        except (TypeError, ValueError):
            continue
        if value == scene_idx:
            return image
        if fallback is None and value in fallback_candidates:
            fallback = image
    return fallback

def _extension_from_content_type(content_type):
    content_type = (content_type or "").lower()
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    if "webp" in content_type:
        return ".webp"
    return ".png"

def _materialize_image(img_data, tmpdir, scene_idx, client, bucket):
    if not img_data:
        return None

    file_key = img_data.get("file_key") or img_data.get("image_file_key")
    if file_key:
        temp_img_path = os.path.join(tmpdir, f"img_{scene_idx}.png").replace("\\", "/")
        print(f"DEBUG_LOG [assemble_video] Downloading R2 image key: {file_key} -> {temp_img_path}")
        download_from_r2(client, bucket, file_key, temp_img_path)
        return temp_img_path

    url = img_data.get("url")
    if not url:
        return None

    if url.startswith("data:image/"):
        header, b64 = url.split(",", 1)
        ext = _extension_from_content_type(header)
        temp_img_path = os.path.join(tmpdir, f"img_{scene_idx}{ext}").replace("\\", "/")
        with open(temp_img_path, "wb") as f:
            f.write(base64.b64decode(b64))
        return temp_img_path

    if url.startswith("http://") or url.startswith("https://"):
        with httpx.Client(timeout=60, follow_redirects=True) as http_client:
            response = http_client.get(url)
            response.raise_for_status()
            ext = _extension_from_content_type(response.headers.get("content-type"))
            temp_img_path = os.path.join(tmpdir, f"img_{scene_idx}{ext}").replace("\\", "/")
            with open(temp_img_path, "wb") as f:
                f.write(response.content)
            return temp_img_path

    return None

def _download_audio_key(audio_key, tmpdir, label, client, bucket):
    if not audio_key:
        return None
    ext = os.path.splitext(audio_key.split("?", 1)[0])[1] or ".mp3"
    path = os.path.join(tmpdir, f"{label}{ext}").replace("\\", "/")
    print(f"DEBUG_LOG [assemble_video] Downloading R2 audio key: {audio_key} -> {path}")
    download_from_r2(client, bucket, audio_key, path)
    return path

# ── Step Functions ────────────────────────────────────────────────────────────

@app.function(image=nvidia_image, gpu="A10G", timeout=1200, memory=8192)
def assemble_video(payload: dict):
    """Pull images + audio from R2, assemble with Ken Burns, upload final video."""
    print(f"DEBUG_LOG [assemble_video] Starting with job_id: {payload.get('job_id')}")
    callback_url = payload["callback_url"]
    callback_secret = payload["callback_secret"]
    r2_config = json.loads(payload.get("r2_config", "{}"))
    client = get_r2_client(r2_config)
    bucket = r2_config.get("bucket_name", "")
    input_key = payload.get("input_key")
    output_folder = payload.get("output_folder", "")
    settings = json.loads(payload.get("settings", "{}"))

    try:
        scenes = settings.get("scenes", [])
        images = settings.get("images", [])
        music_file_key = payload.get("music_file_key") or settings.get("music_file_key")
        encoder = detect_encoder()
        codec, extra_args = ("h264_nvenc", "-preset p4 -rc vbr -cq 23") if encoder == "h264_nvenc" else ("libx264", "-preset fast -crf 23")

        print(f"DEBUG_LOG [assemble_video] Parsed configuration. scenes: {len(scenes)}, images: {len(images)}, music_key: {music_file_key}, codec: {codec}")
        if not scenes:
            raise ValueError("No scenes were provided for SmartVideo assembly.")
        print(f"DEBUG_LOG [assemble_video] Creating temporary directory...")
        with tempfile.TemporaryDirectory() as tmpdir:
            normalized_clips = []
            print(f"DEBUG_LOG [assemble_video] Temp directory created at: {tmpdir}")

            # Process each scene
            for idx, scene in enumerate(scenes):
                scene_idx = int(scene.get("index", idx))
                duration = float(scene.get("durationSeconds", 5.0))
                width, height = 1080, 1920

                img_data = _matching_scene_image(images, scene_idx, idx)
                temp_img_path = _materialize_image(img_data, tmpdir, scene_idx, client, bucket)
                scene_audio_path = _download_audio_key(
                    scene.get("audio_file_key"),
                    tmpdir,
                    f"voice_scene_{scene_idx}",
                    client,
                    bucket
                )
                norm_clip_path = os.path.join(tmpdir, f"norm_clip_{scene_idx}.mp4").replace("\\", "/")
                print(f"DEBUG_LOG [assemble_video] Processing scene {scene_idx} (duration: {duration}s). Output clip path: {norm_clip_path}")

                if temp_img_path:
                    cmd = [
                        "ffmpeg", "-y", "-loop", "1", "-t", f"{duration:.3f}", "-i", temp_img_path,
                    ]
                    if scene_audio_path:
                        cmd += ["-i", scene_audio_path]
                    else:
                        cmd += ["-f", "lavfi", "-t", f"{duration:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
                    cmd += [
                        "-vf", f"scale={width}:{height},setsar=1,format=yuv420p",
                        "-map", "0:v", "-map", "1:a",
                        "-c:v", codec
                    ] + extra_args.split() + ["-c:a", "aac", "-b:a", "192k", "-shortest", "-r", "25", norm_clip_path]
                    print(f"DEBUG_LOG [assemble_video] Running image loop ffmpeg command: {' '.join(cmd)}")
                    subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
                else:
                    print(f"DEBUG_LOG [assemble_video] No image for scene {scene_idx}. Generating black background clip.")
                    cmd = [
                        "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r=25",
                        "-t", f"{duration:.3f}"
                    ]
                    if scene_audio_path:
                        cmd += ["-i", scene_audio_path]
                    else:
                        cmd += ["-f", "lavfi", "-t", f"{duration:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
                    cmd += [
                        "-map", "0:v", "-map", "1:a", "-c:v", codec
                    ] + extra_args.split() + ["-c:a", "aac", "-b:a", "192k", "-shortest", "-pix_fmt", "yuv420p", norm_clip_path]
                    print(f"DEBUG_LOG [assemble_video] Running lavfi black background command: {' '.join(cmd)}")
                    subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)

                print(f"DEBUG_LOG [assemble_video] Clip {scene_idx} generated successfully. size: {os.path.getsize(norm_clip_path)} bytes")
                normalized_clips.append(norm_clip_path)

            # Concat clips
            concat_list = os.path.join(tmpdir, "concat.txt").replace("\\", "/")
            print(f"DEBUG_LOG [assemble_video] Creating concat list at: {concat_list} for {len(normalized_clips)} clips")
            if not normalized_clips:
                raise ValueError("SmartVideo assembly produced no scene clips.")
            with open(concat_list, "w") as f:
                for clip in normalized_clips:
                    f.write(f"file '{clip.replace(chr(92), '/')}'\n")

            merged = os.path.join(tmpdir, "merged.mp4").replace("\\", "/")
            cmd_concat = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", merged]
            print(f"DEBUG_LOG [assemble_video] Running concat command: {' '.join(cmd_concat)}")
            subprocess.run(cmd_concat, capture_output=True, text=True, timeout=180, check=True)
            print(f"DEBUG_LOG [assemble_video] Merged video generated successfully. size: {os.path.getsize(merged)} bytes")

            # Mix music
            final = os.path.join(tmpdir, "final.mp4").replace("\\", "/")
            music_path = None
            if music_file_key:
                music_path = os.path.join(tmpdir, "music.mp3").replace("\\", "/")
                print(f"DEBUG_LOG [assemble_video] Music file key: {music_file_key}. Downloading to: {music_path}")
                download_from_r2(client, bucket, music_file_key, music_path)
                print(f"DEBUG_LOG [assemble_video] Music download complete. size: {os.path.getsize(music_path)} bytes")

            if music_path:
                cmd_mix = ["ffmpeg", "-y", "-i", merged, "-i", music_path,
                     "-filter_complex", "[1:a]volume=0.25[music];[0:a][music]amix=inputs=2:duration=first[aout]",
                     "-map", "0:v", "-map", "[aout]", "-shortest", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", final]
                print(f"DEBUG_LOG [assemble_video] Running audio mix command: {' '.join(cmd_mix)}")
                try:
                    subprocess.run(cmd_mix, capture_output=True, text=True, timeout=240, check=True)
                except subprocess.CalledProcessError as mix_error:
                    print(f"DEBUG_LOG [assemble_video] Music mix failed; falling back to narration-only video: {mix_error.stderr}")
                    cmd_copy = ["ffmpeg", "-y", "-i", merged, "-c", "copy", final]
                    subprocess.run(cmd_copy, capture_output=True, text=True, timeout=120, check=True)
            else:
                cmd_copy = ["ffmpeg", "-y", "-i", merged, "-c", "copy", final]
                print(f"DEBUG_LOG [assemble_video] Running raw copy command (no music): {' '.join(cmd_copy)}")
                subprocess.run(cmd_copy, check=True)

            print(f"DEBUG_LOG [assemble_video] Final video generated at: {final}. size: {os.path.getsize(final)} bytes")
            if output_folder and not output_folder.endswith("/"):
                output_folder = f"{output_folder}/"
            output_key = f"{output_folder}final_{payload['job_id']}.mp4"
            print(f"DEBUG_LOG [assemble_video] Uploading final video to R2 key: {output_key}")
            upload_to_r2(client, bucket, output_key, final, "video/mp4")
            print("DEBUG_LOG [assemble_video] Upload complete.")

        print(f"DEBUG_LOG [assemble_video] Sending success callback to: {callback_url}")
        send_callback(callback_url, callback_secret, {
            "success": True,
            "outputFilesJson": json.dumps({"FinalVideoKey": output_key})
        })
        print("DEBUG_LOG [assemble_video] Success callback sent.")
    except Exception as e:
        import traceback
        print("DEBUG_LOG [assemble_video] EXCEPTION OCCURRED:")
        traceback.print_exc()
        print(f"DEBUG_LOG [assemble_video] Sending failure callback to: {callback_url} with error: {str(e)}")
        send_callback(callback_url, callback_secret, {
            "success": False,
            "errorMessage": str(e)
        })
        print("DEBUG_LOG [assemble_video] Failure callback sent.")

@app.function(timeout=300, memory=4096)
def generate_thumbnails(payload: dict):
    """Extract frame thumbnails from assembled video."""
    print(f"DEBUG_LOG [generate_thumbnails] Starting with job_id: {payload.get('job_id')}")
    callback_url = payload["callback_url"]
    callback_secret = payload["callback_secret"]
    r2_config = json.loads(payload.get("r2_config", "{}"))
    client = get_r2_client(r2_config)
    bucket = r2_config.get("bucket_name", "")
    input_key = payload.get("input_key")
    output_folder = payload.get("output_folder", "")

    print(f"DEBUG_LOG [generate_thumbnails] Parsed configuration. input_key: {input_key}")
    try:
        thumb_keys = []
        print("DEBUG_LOG [generate_thumbnails] Creating temporary directory...")
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4").replace("\\", "/")
            print(f"DEBUG_LOG [generate_thumbnails] Downloading assembled video from R2: {input_key} -> {video_path}")
            download_from_r2(client, bucket, input_key, video_path)
            print(f"DEBUG_LOG [generate_thumbnails] Download complete. size: {os.path.getsize(video_path)} bytes")

            timestamps = [1, 5, 10, 15, 20]
            for i, ts in enumerate(timestamps):
                thumb_path = os.path.join(tmpdir, f"thumb_{i}.jpg").replace("\\", "/")
                cmd = ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path, "-vframes", "1", "-q:v", "2", thumb_path]
                print(f"DEBUG_LOG [generate_thumbnails] Extracting thumbnail {i} at {ts}s. Command: {' '.join(cmd)}")
                subprocess.run(cmd, capture_output=True, text=True, timeout=55, check=True)
                
                key = f"{output_folder}thumb_{i}_{payload['job_id']}.jpg"
                print(f"DEBUG_LOG [generate_thumbnails] Uploading thumbnail {i} to R2 key: {key}")
                upload_to_r2(client, bucket, key, thumb_path, "image/jpeg")
                thumb_keys.append(key)

        print(f"DEBUG_LOG [generate_thumbnails] Sending success callback to: {callback_url}")
        send_callback(callback_url, callback_secret, {
            "success": True,
            "outputFilesJson": json.dumps({"ThumbnailsJson": json.dumps(thumb_keys)})
        })
        print("DEBUG_LOG [generate_thumbnails] Success callback sent.")
    except Exception as e:
        import traceback
        print("DEBUG_LOG [generate_thumbnails] EXCEPTION OCCURRED:")
        traceback.print_exc()
        print(f"DEBUG_LOG [generate_thumbnails] Sending failure callback to: {callback_url} with error: {str(e)}")
        send_callback(callback_url, callback_secret, {
            "success": False,
            "errorMessage": str(e)
        })
        print("DEBUG_LOG [generate_thumbnails] Failure callback sent.")

@app.function(timeout=300, memory=4096)
def generate_shorts(payload: dict):
    """Cut shorts from assembled video."""
    print(f"DEBUG_LOG [generate_shorts] Starting with job_id: {payload.get('job_id')}")
    callback_url = payload["callback_url"]
    callback_secret = payload["callback_secret"]
    r2_config = json.loads(payload.get("r2_config", "{}"))
    client = get_r2_client(r2_config)
    bucket = r2_config.get("bucket_name", "")
    input_key = payload.get("input_key")
    output_folder = payload.get("output_folder", "")
    settings = json.loads(payload.get("settings", "{}"))

    print(f"DEBUG_LOG [generate_shorts] Parsed configuration. input_key: {input_key}")
    try:
        short_keys = []
        encoder = detect_encoder()
        print("DEBUG_LOG [generate_shorts] Creating temporary directory...")
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4").replace("\\", "/")
            print(f"DEBUG_LOG [generate_shorts] Downloading assembled video from R2: {input_key} -> {video_path}")
            download_from_r2(client, bucket, input_key, video_path)
            print(f"DEBUG_LOG [generate_shorts] Download complete. size: {os.path.getsize(video_path)} bytes")

            scenes = settings.get("scenes", [])
            count = min(int(settings.get("short_count", 3)), len(scenes))
            max_dur = float(settings.get("max_duration", 15))
            print(f"DEBUG_LOG [generate_shorts] Cutting {count} shorts (max_duration: {max_dur}s) from {len(scenes)} scenes")

            current_time = 0.0
            for i in range(count):
                dur = min(float(scenes[i].get("durationSeconds", 5)), max_dur)
                out = os.path.join(tmpdir, f"short_{i+1}.mp4").replace("\\", "/")
                cmd = ["ffmpeg", "-y", "-ss", f"{current_time:.3f}", "-t", f"{dur:.3f}",
                     "-i", video_path,
                     "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
                     "-c:v", encoder, "-c:a", "aac", out]
                print(f"DEBUG_LOG [generate_shorts] Cutting short {i+1} at {current_time}s (duration: {dur}s). Command: {' '.join(cmd)}")
                subprocess.run(cmd, capture_output=True, text=True, timeout=90, check=True)
                
                k = f"{output_folder}short_{i+1}_{payload['job_id']}.mp4"
                print(f"DEBUG_LOG [generate_shorts] Uploading short {i+1} to R2 key: {k}")
                upload_to_r2(client, bucket, k, out, "video/mp4")
                short_keys.append(k)
                current_time += float(scenes[i].get("durationSeconds", 5))

        print(f"DEBUG_LOG [generate_shorts] Sending success callback to: {callback_url}")
        send_callback(callback_url, callback_secret, {
            "success": True,
            "outputFilesJson": json.dumps({"ShortsJson": json.dumps(short_keys)})
        })
        print("DEBUG_LOG [generate_shorts] Success callback sent.")
    except Exception as e:
        import traceback
        print("DEBUG_LOG [generate_shorts] EXCEPTION OCCURRED:")
        traceback.print_exc()
        print(f"DEBUG_LOG [generate_shorts] Sending failure callback to: {callback_url} with error: {str(e)}")
        send_callback(callback_url, callback_secret, {
            "success": False,
            "errorMessage": str(e)
        })
        print("DEBUG_LOG [generate_shorts] Failure callback sent.")

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
        "AssemblingVideo":      assemble_video,
        "GeneratingThumbnails": generate_thumbnails,
        # "GeneratingAiVideo" — not yet implemented
        "GeneratingShorts":     generate_shorts,
    }
    fn = function_map.get(step_name)
    if fn is None:
        raise fastapi.HTTPException(status_code=400, detail=f"Unknown step: {step_name}")
    fn.spawn(body)
    return {"dispatched": True}

