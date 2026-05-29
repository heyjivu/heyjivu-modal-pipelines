"""
heyjivu Modal Social Post Render Pipeline
App Name: aura-social-post-pipeline

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

app = modal.App("aura-social-post-pipeline", image=image)

@app.function()
@modal.fastapi_endpoint(method="GET")
def health():
    return {"status": "ok", "app": "aura-social-post-pipeline"}

# ── Step Functions ────────────────────────────────────────────────────────────

@app.function(image=nvidia_image, gpu="A10G", timeout=1200, memory=8192)
def mix_audio_video(payload: dict):
    """Pull assets, mix audio/video layers, upload rendered post, callback."""
    print(f"DEBUG_LOG [mix_audio_video] Starting with job_id: {payload.get('job_id')}")
    callback_url = payload["callback_url"]
    callback_secret = payload["callback_secret"]
    r2_config = json.loads(payload.get("r2_config", "{}"))
    client = get_r2_client(r2_config)
    bucket = r2_config.get("bucket_name", "")
    input_key = payload.get("input_key")
    output_folder = payload.get("output_folder", "")
    settings = json.loads(payload.get("settings", "{}"))

    try:
        encoder = detect_encoder()
        codec, extra_args = ("h264_nvenc", "-preset p4 -rc vbr -cq 23") if encoder == "h264_nvenc" else ("libx264", "-preset fast -crf 23")

        media_items = settings.get("media_items", [])
        audio_items = settings.get("audio_items", [])
        width, height = 1080, 1920

        print(f"DEBUG_LOG [mix_audio_video] Parsed configuration. media_items: {len(media_items)}, audio_items: {len(audio_items)}, encoder: {encoder}")
        print("DEBUG_LOG [mix_audio_video] Creating temporary directory...")
        with tempfile.TemporaryDirectory() as tmpdir:
            normalized_clips = []
            print(f"DEBUG_LOG [mix_audio_video] Temp directory created at: {tmpdir}")

            for i, item in enumerate(media_items):
                ext = item.get("file_ext", "mp4")
                inp = os.path.join(tmpdir, f"input_{i}.{ext}").replace("\\", "/")
                print(f"DEBUG_LOG [mix_audio_video] Downloading media item {i} (key: {item['file_key']}) to: {inp}")
                download_from_r2(client, bucket, item["file_key"], inp)
                print(f"DEBUG_LOG [mix_audio_video] Download complete. size: {os.path.getsize(inp)} bytes")

                norm = os.path.join(tmpdir, f"norm_{i}.mp4").replace("\\", "/")
                mtype = item.get("media_type", "video")
                if mtype == "image":
                    dur = float(item.get("duration", 5.0))
                    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", inp,
                         "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
                         "-c:v", codec] + extra_args.split() + ["-t", f"{dur:.3f}", "-r", "30", norm]
                    print(f"DEBUG_LOG [mix_audio_video] Running image-to-video command for item {i}: {' '.join(cmd)}")
                    subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
                else:
                    cmd = ["ffmpeg", "-y", "-i", inp,
                         "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
                         "-c:v", codec] + extra_args.split() + ["-c:a", "aac", norm]
                    print(f"DEBUG_LOG [mix_audio_video] Running video normalization command for item {i}: {' '.join(cmd)}")
                    subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
                
                print(f"DEBUG_LOG [mix_audio_video] Normalized clip {i} generated successfully. size: {os.path.getsize(norm)} bytes")
                normalized_clips.append(norm)

            # Concat clips
            concat_list = os.path.join(tmpdir, "concat.txt").replace("\\", "/")
            print(f"DEBUG_LOG [mix_audio_video] Creating concat list at: {concat_list} for {len(normalized_clips)} clips")
            with open(concat_list, "w") as f:
                for clip in normalized_clips:
                    f.write(f"file '{clip.replace(chr(92), '/')}'\n")

            merged = os.path.join(tmpdir, "merged.mp4").replace("\\", "/")
            cmd_concat = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", merged]
            print(f"DEBUG_LOG [mix_audio_video] Running concat command: {' '.join(cmd_concat)}")
            subprocess.run(cmd_concat, capture_output=True, text=True, timeout=300, check=True)
            print(f"DEBUG_LOG [mix_audio_video] Merged video generated successfully. size: {os.path.getsize(merged)} bytes")

            # Mix audio — use all available audio tracks
            final = os.path.join(tmpdir, "final.mp4").replace("\\", "/")
            if audio_items:
                audio_paths = []
                for idx, audio in enumerate(audio_items):
                    aud_file = audio.get("file_key")
                    if aud_file:
                        aud_local = os.path.join(tmpdir, f"audio_{idx}.mp3").replace("\\", "/")
                        print(f"DEBUG_LOG [mix_audio_video] Downloading audio item {idx} (key: {aud_file}) to: {aud_local}")
                        download_from_r2(client, bucket, aud_file, aud_local)
                        print(f"DEBUG_LOG [mix_audio_video] Download complete. size: {os.path.getsize(aud_local)} bytes")
                        audio_paths.append(aud_local)

                if audio_paths:
                    if len(audio_paths) == 1:
                        cmd_mix = ["ffmpeg", "-y", "-i", merged, "-i", audio_paths[0],
                             "-filter_complex", "[1:a]volume=1.0[a1];[0:a][a1]amix=inputs=2:duration=first[aout]",
                             "-map", "0:v", "-map", "[aout]", "-shortest", "-c:v", "copy", "-c:a", "aac", final]
                        print(f"DEBUG_LOG [mix_audio_video] Running single audio mix command: {' '.join(cmd_mix)}")
                        subprocess.run(cmd_mix, capture_output=True, text=True, timeout=300, check=True)
                    else:
                        # Build amix with all audio tracks
                        filter_inputs = "".join(f"[{i+1}:a]" for i in range(len(audio_paths)))
                        filter_labels = ";".join(
                            f"[{i+1}:a]volume={audio.get('volume', 1.0)}[a{i}]"
                            for i, audio in enumerate(audio_items) if audio.get("file_key")
                        )
                        amix_inputs = ":".join(f"[a{i}]" for i in range(len(audio_paths)))
                        filter_complex = f"{filter_labels};{amix_inputs}amix=inputs={len(audio_paths)}:duration=first[aout]"
                        cmd_mix_multi = ["ffmpeg", "-y", "-i", merged] + \
                            [item for pair in [("-i", p) for p in audio_paths] for item in pair] + \
                            ["-filter_complex", filter_complex,
                             "-map", "0:v", "-map", "[aout]", "-shortest", "-c:v", "copy", "-c:a", "aac", final]
                        print(f"DEBUG_LOG [mix_audio_video] Running multi-audio mix command: {' '.join(cmd_mix_multi)}")
                        subprocess.run(cmd_mix_multi, capture_output=True, text=True, timeout=300, check=True)
                else:
                    cmd_copy = ["ffmpeg", "-y", "-i", merged, "-c", "copy", final]
                    print(f"DEBUG_LOG [mix_audio_video] Running copy command (no active audio): {' '.join(cmd_copy)}")
                    subprocess.run(cmd_copy, check=True)
            else:
                cmd_copy = ["ffmpeg", "-y", "-i", merged, "-c", "copy", final]
                print(f"DEBUG_LOG [mix_audio_video] Running copy command (no audio items): {' '.join(cmd_copy)}")
                subprocess.run(cmd_copy, check=True)

            print(f"DEBUG_LOG [mix_audio_video] Final video generated at: {final}. size: {os.path.getsize(final)} bytes")
            output_key = f"{output_folder}rendered_{payload['job_id']}.mp4"
            print(f"DEBUG_LOG [mix_audio_video] Uploading final social video to R2 key: {output_key}")
            upload_to_r2(client, bucket, output_key, final, "video/mp4")
            print("DEBUG_LOG [mix_audio_video] Upload complete.")

        print(f"DEBUG_LOG [mix_audio_video] Sending success callback to: {callback_url}")
        send_callback(callback_url, callback_secret, {
            "success": True,
            "outputFilesJson": json.dumps({"RenderedPostKey": output_key})
        })
        print("DEBUG_LOG [mix_audio_video] Success callback sent.")
    except Exception as e:
        import traceback
        print("DEBUG_LOG [mix_audio_video] EXCEPTION OCCURRED:")
        traceback.print_exc()
        print(f"DEBUG_LOG [mix_audio_video] Sending failure callback to: {callback_url} with error: {str(e)}")
        send_callback(callback_url, callback_secret, {
            "success": False,
            "errorMessage": str(e)
        })
        print("DEBUG_LOG [mix_audio_video] Failure callback sent.")

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
        "MixAudioVideo": mix_audio_video,
    }
    fn = function_map.get(step_name)
    if fn is None:
        return {"error": f"Unknown step: {step_name}"}
    fn.spawn(body)
    return {"dispatched": True}

