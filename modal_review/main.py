"""
heyjivu Modal Review Pipeline
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

from modal_common import with_retry, get_r2_client, download_from_r2, upload_to_r2

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "fonts-dejavu-core")
    .pip_install("fastapi", "httpx", "boto3", "botocore", "google-api-python-client", "google-auth", "google-auth-oauthlib", "google-auth-httplib2")
        .add_local_dir("./modal_common", "/root/modal_common")
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

# Review Template FFmpeg Helpers

def _as_dict(value):
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}

def _first_value(*values):
    for value in values:
        if value is not None and value != "":
            return value
    return None

def _template_edit(payload: dict) -> dict:
    context = _as_dict(payload.get("template_edit"))
    edit = _as_dict(context.get("edit") or context.get("Edit") or context)
    return edit if edit else {}

def _nested(edit: dict, *keys):
    for key in keys:
        value = edit.get(key)
        if value is not None:
            return _as_dict(value)
    return {}

def _flatten_template_config(edit: dict) -> dict:
    template_payload = _nested(edit, "templatePayloadJson", "template_payload_json", "templatePayload", "payload", "config")
    overlay = _nested(edit, "overlaySettingsJson", "overlay_settings_json", "overlay", "overlaySettings")
    filters = _nested(edit, "filterSettings", "filter_settings", "filters")
    effects = _nested(edit, "effectSettings", "effect_settings", "effects")
    audio = _nested(edit, "audioOptionsJson", "audio_options_json", "audioOptions", "audio")

    merged = {}
    for source in (template_payload, overlay, filters, effects, audio):
        merged.update(source)

    for key in (
        "templateId", "templateName", "templateType", "templateSource",
        "effectId", "effectName", "audioConflictStrategy", "soundtrackId"
    ):
        if edit.get(key) is not None:
            merged[key] = edit.get(key)

    return merged

def _target_dimensions(config: dict, asset_type: str | None):
    aspect = str(_first_value(
        config.get("aspectRatio"),
        config.get("aspect_ratio"),
        config.get("outputAspectRatio"),
        "9:16" if asset_type == "shorts" else None
    ) or "").strip()

    if aspect in ("9:16", "vertical", "reel", "short"):
        return 1080, 1920
    if aspect in ("4:5", "portrait"):
        return 1080, 1350
    if aspect in ("1:1", "square"):
        return 1080, 1080
    return 1920, 1080

def _safe_color(value, fallback="white"):
    if not value:
        return fallback
    text = str(value).strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#@._:-")
    if not all(ch in allowed for ch in text) or len(text) > 40:
        return fallback
    return "0x" + text[1:] if text.startswith("#") else text

def _escape_drawtext(text):
    value = str(text or "")[:220]
    return (
        value
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
        .replace("\n", " ")
        .replace("\r", " ")
    )

def _filter_for(filter_id):
    normalized = str(filter_id or "").strip().lower()
    if normalized in ("bw", "mono", "noir", "blackwhite"):
        return "hue=s=0"
    if normalized == "sepia":
        return "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131"
    if normalized == "vintage":
        return "eq=contrast=1.08:brightness=0.02:saturation=1.28,colorbalance=rs=.08:gs=.03:bs=-.08"
    if normalized == "cyberpunk":
        return "eq=contrast=1.18:saturation=1.75,hue=h=28"
    if normalized == "warm":
        return "eq=brightness=0.02:saturation=1.16,colorbalance=rs=.07:gs=.02:bs=-.05"
    if normalized == "clean":
        return "eq=contrast=1.04:saturation=1.05,unsharp=5:5:0.45"
    if normalized == "punchy":
        return "eq=contrast=1.18:saturation=1.28,unsharp=5:5:0.55"
    return None

def _ffprobe_duration(path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", path],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return max(0.0, float(result.stdout.strip()))
    except Exception:
        return 0.0

def _has_audio_stream(path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index", "-of", "csv=p=0", path],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False

def _resolve_external_audio(config: dict, tmpdir: str, r2_client=None, r2_bucket: str = ""):
    audio_url = _first_value(config.get("audioUrl"), config.get("audio_url"), config.get("previewUrl"), config.get("soundtrackUrl"))
    audio_key = _first_value(config.get("audioFileKey"), config.get("audio_file_key"), config.get("fileKey"), config.get("soundtrackFileKey"))
    if audio_key and r2_client and r2_bucket:
        path = os.path.join(tmpdir, "template_audio.mp3").replace("\\", "/")
        download_from_r2(r2_client, r2_bucket, audio_key, path)
        return path
    if audio_url and str(audio_url).startswith(("http://", "https://")):
        path = os.path.join(tmpdir, "template_audio.mp3").replace("\\", "/")
        with httpx.stream("GET", str(audio_url), timeout=60, follow_redirects=True) as response:
            response.raise_for_status()
            with open(path, "wb") as fh:
                for chunk in response.iter_bytes():
                    if chunk:
                        fh.write(chunk)
        return path
    return None

def _build_video_filters(config: dict, edit: dict, asset_type: str | None, input_path: str, external_audio_path: str | None):
    width, height = _target_dimensions(config, asset_type)
    filters = [
        f"scale={width}:{height}:force_original_aspect_ratio=increase",
        f"crop={width}:{height}",
        "setsar=1",
        "format=yuv420p",
    ]

    chosen_filter = _first_value(config.get("filter"), config.get("filterId"), config.get("effectId"), edit.get("effectId"))
    ff_filter = _filter_for(chosen_filter)
    if ff_filter:
        filters.append(ff_filter)

    template_id = str(_first_value(config.get("templateId"), edit.get("templateId")) or "").lower()
    frame_enabled = bool(config.get("frame") or config.get("border") or "frame" in template_id or "spotlight" in template_id)
    if frame_enabled:
        border_color = _safe_color(config.get("borderColor") or "#ffffff@0.75")
        filters.append(f"drawbox=x=24:y=24:w=iw-48:h=ih-48:color={border_color}:t=6")

    if config.get("progressBar") is not False and (asset_type == "shorts" or "short" in template_id or "reel" in template_id):
        progress_color = _safe_color(config.get("progressColor") or "#10b981")
        filters.append(f"drawbox=x=0:y=ih-14:w='iw*t/{max(_ffprobe_duration(input_path), 1.0):.3f}':h=14:color={progress_color}:t=fill")

    watermark = _first_value(config.get("watermark"), config.get("brandText"), edit.get("watermark"))
    if watermark:
        filters.append(
            "drawtext="
            f"text='{_escape_drawtext(watermark)}':"
            "x=40:y=h-th-48:fontsize=34:"
            "fontcolor=white@0.92:"
            "box=1:boxcolor=black@0.36:boxborderw=16"
        )

    lower_third = _first_value(config.get("lowerThird"), config.get("headline"), config.get("hookText"))
    if lower_third:
        filters.append("drawbox=x=40:y=h*0.68:w=iw-80:h=150:color=black@0.48:t=fill")
        filters.append(
            "drawtext="
            f"text='{_escape_drawtext(lower_third)}':"
            "x=70:y=h*0.68+36:fontsize=44:fontcolor=white"
        )

    cta = _first_value(config.get("cta"), config.get("callToAction"), config.get("call_to_action"))
    if cta:
        filters.append(
            "drawtext="
            f"text='{_escape_drawtext(cta)}':"
            "x=(w-text_w)/2:y=h-th-120:fontsize=38:"
            f"fontcolor=white:box=1:boxcolor={_safe_color('#10b981@0.72')}:boxborderw=18"
        )

    for element in config.get("canvasElements") or []:
        if not isinstance(element, dict) or element.get("type") != "text" or not element.get("content"):
            continue
        x_pct = max(0, min(95, float(element.get("x", 5)))) / 100.0
        y_pct = max(0, min(95, float(element.get("y", 80)))) / 100.0
        size = max(12, min(96, int(element.get("fontSize", 28))))
        color = _safe_color(element.get("color") or "white")
        filters.append(
            "drawtext="
            f"text='{_escape_drawtext(element.get('content'))}':"
            f"x=w*{x_pct:.4f}:y=h*{y_pct:.4f}:fontsize={size}:fontcolor={color}:"
            "box=1:boxcolor=black@0.24:boxborderw=10"
        )

    strategy = str(_first_value(config.get("audioConflictStrategy"), edit.get("audioConflictStrategy")) or "").lower()
    if external_audio_path and strategy == "extendvideo":
        video_duration = _ffprobe_duration(input_path)
        audio_duration = _ffprobe_duration(external_audio_path)
        if audio_duration > video_duration > 0:
            filters.append(f"tpad=stop_mode=clone:stop_duration={audio_duration - video_duration:.3f}")

    return ",".join(filters)

def apply_template_render(input_path: str, output_path: str, template_edit: dict, asset_type: str | None = None, tmpdir: str | None = None, r2_client=None, r2_bucket: str = "") -> bool:
    if not template_edit:
        return False

    config = _flatten_template_config(template_edit)
    if not config and not template_edit:
        return False

    tmpdir = tmpdir or os.path.dirname(output_path)
    strategy = str(_first_value(config.get("audioConflictStrategy"), template_edit.get("audioConflictStrategy")) or "keepOriginal").lower()
    external_audio_path = None if strategy == "keeporiginal" else _resolve_external_audio(config, tmpdir, r2_client, r2_bucket)
    vf = _build_video_filters(config, template_edit, asset_type, input_path, external_audio_path)

    cmd = ["ffmpeg", "-y", "-i", input_path]
    if external_audio_path:
        if strategy == "repeataudio":
            cmd.extend(["-stream_loop", "-1"])
        cmd.extend(["-i", external_audio_path])

    has_original_audio = _has_audio_stream(input_path)
    filter_parts = [f"[0:v]{vf}[vout]"]
    map_args = ["-map", "[vout]"]

    if external_audio_path:
        volume = float(_first_value(config.get("soundtrackVolume"), config.get("volume"), 50) or 50)
        volume = volume / 100.0 if volume > 1 else volume
        duration_mode = "longest" if strategy == "extendvideo" else "first"
        if has_original_audio:
            filter_parts.append(f"[1:a]volume={volume:.3f}[music];[0:a][music]amix=inputs=2:duration={duration_mode}:dropout_transition=2[aout]")
        else:
            filter_parts.append(f"[1:a]volume={volume:.3f}[aout]")
        map_args.extend(["-map", "[aout]"])
    else:
        map_args.extend(["-map", "0:a?"])

    cmd.extend([
        "-filter_complex", ";".join(filter_parts),
        *map_args,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
    ])
    if external_audio_path and strategy in ("trimaudio", "repeataudio"):
        cmd.append("-shortest")
    cmd.append(output_path)

    print(f"DEBUG_LOG [apply_template_render] Running FFmpeg template command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"DEBUG_LOG [apply_template_render] FFmpeg template render failed: {result.stderr}")
        raise RuntimeError(f"FFmpeg template render failed:\n{result.stderr}")

    return True

def parse_r2_config(payload: dict):
    config = payload.get("r2_config", {})
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except Exception:
            config = {}
    return config if isinstance(config, dict) else {}

def delete_r2_object(client, bucket: str, key: str):
    try:
        client.delete_object(Bucket=bucket, Key=key)
    except Exception as e:
        print(f"WARNING: Failed to delete R2 object {key}: {e}")

def move_r2_prefix(client, bucket: str, source_prefix: str, target_prefix: str):
    source = source_prefix.strip("/")
    target = target_prefix.strip("/")
    if not source or not target:
        raise ValueError("source and target prefixes are required for R2 finalize.")
    if source == target or source.startswith(target + "/"):
        return

    paginator = client.get_paginator("list_objects_v2")
    moved = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=source + "/"):
        for obj in page.get("Contents", []):
            old_key = obj["Key"]
            new_key = f"{target}/{old_key[len(source):].lstrip('/')}"
            client.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": old_key}, Key=new_key)
            client.delete_object(Bucket=bucket, Key=old_key)
            moved += 1
    print(f"DEBUG_LOG [move_r2_prefix] Moved {moved} objects from {source} to {target}")

# ── Orchestrated Worker Function ──────────────────────────────────────────────

@app.function(timeout=600, memory=4096)
def process_review_job(payload: dict):
    video_id = payload.get("video_id")
    is_finalize = payload.get("is_finalize", False)
    storage_type = str(payload.get("storage_type") or "drive").lower()
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
    template_edit = _template_edit(payload)
    template_context = _as_dict(payload.get("template_edit"))
    asset_type = str(template_context.get("assetType") or template_context.get("asset_type") or "").lower() or None
    
    oauth = payload.get("oauth_credentials", {})
    client_id = oauth.get("client_id")
    client_secret = oauth.get("client_secret")
    refresh_token = oauth.get("refresh_token")
    
    gemini_api_key = payload.get("gemini_api_key")
    callback_url = payload.get("callback_url")
    callback_secret = payload.get("callback_secret")
    r2_client = None
    r2_bucket = ""

    print(f"DEBUG_LOG [process_review_job] Started for video_id: {video_id}. storage_type: {storage_type}, is_finalize: {is_finalize}, generate_video: {generate_video}, generate_thumbnail: {generate_thumbnail}, generate_description: {generate_description}, has_template_edit: {bool(template_edit)}")
    try:
        service = None
        if storage_type == "r2":
            r2_config = parse_r2_config(payload)
            r2_client = get_r2_client(r2_config)
            r2_bucket = r2_config.get("bucket_name", "")
            if not r2_client or not r2_bucket:
                raise ValueError("R2 storage credentials are not configured for review rendering.")
            print("DEBUG_LOG [process_review_job] R2 client built successfully.")
        else:
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

            if template_edit and video_file_id:
                with tempfile.TemporaryDirectory() as tmpdir:
                    input_video_path = os.path.join(tmpdir, "input.mp4").replace("\\", "/")
                    rendered_video_path = os.path.join(tmpdir, "template_render.mp4").replace("\\", "/")
                    if storage_type == "r2":
                        print(f"DEBUG_LOG [process_review_job] Downloading R2 video {video_file_id} for finalize template render...")
                        download_from_r2(r2_client, r2_bucket, video_file_id, input_video_path)
                    else:
                        print(f"DEBUG_LOG [process_review_job] Downloading Drive video {video_file_id} for finalize template render...")
                        download_file(service, video_file_id, input_video_path)

                    if apply_template_render(input_video_path, rendered_video_path, template_edit, asset_type, tmpdir, r2_client, r2_bucket):
                        if storage_type == "r2":
                            rendered_key = f"{folder_id.rstrip('/')}/video_template_rendered.mp4"
                            upload_to_r2(r2_client, r2_bucket, rendered_key, rendered_video_path, "video/mp4")
                            if video_file_id != rendered_key:
                                delete_r2_object(r2_client, r2_bucket, video_file_id)
                            video_file_id = rendered_key
                        else:
                            new_vid_id = upload_file(service, rendered_video_path, "video_template_rendered.mp4", folder_id, "video/mp4")
                            if video_file_id and video_file_id != new_vid_id:
                                delete_file(service, video_file_id)
                            video_file_id = new_vid_id
                        print(f"DEBUG_LOG [process_review_job] Finalize template render uploaded. New video id/key: {video_file_id}")

            print(f"DEBUG_LOG [process_review_job] Finalizing review. Moving folder {folder_id} to parent {target_folder_id}")
            if storage_type == "r2":
                move_r2_prefix(r2_client, r2_bucket, folder_id, target_folder_id)
            else:
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
        need_template_render = bool(template_edit)
        need_download = (generate_video and start_time is not None and end_time is not None) or generate_thumbnail or need_template_render
        print(f"DEBUG_LOG [process_review_job] Need file download: {need_download}")
        
        if need_download:
            if not video_file_id:
                raise ValueError("video_file_id is required for video or thumbnail generation.")

            print("DEBUG_LOG [process_review_job] Creating temporary directory...")
            with tempfile.TemporaryDirectory() as tmpdir:
                input_video_path = os.path.join(tmpdir, "input.mp4")
                print(f"DEBUG_LOG [process_review_job] Temp directory: {tmpdir}")
                print(f"DEBUG_LOG [process_review_job] Downloading original video {video_file_id} to {input_video_path}...")
                if storage_type == "r2":
                    download_from_r2(r2_client, r2_bucket, video_file_id, input_video_path)
                else:
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

                # Apply Review template edit metadata as a deterministic FFmpeg render.
                if need_template_render:
                    templated_video_path = os.path.join(tmpdir, "template_rendered.mp4").replace("\\", "/")
                    print(f"DEBUG_LOG [process_review_job] Applying Review template edit to {source_video_path} -> {templated_video_path}...")
                    apply_template_render(source_video_path, templated_video_path, template_edit, asset_type, tmpdir, r2_client, r2_bucket)
                    source_video_path = templated_video_path
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
                    if storage_type == "r2":
                        new_thumb_id = f"{folder_id.rstrip('/')}/thumbnail.png"
                        upload_to_r2(r2_client, r2_bucket, new_thumb_id, thumb_path, "image/png")
                    else:
                        new_thumb_id = upload_file(service, thumb_path, "thumbnail.png", folder_id, "image/png")
                    print(f"DEBUG_LOG [process_review_job] Thumbnail uploaded. New ID: {new_thumb_id}")
                    
                    # Delete old thumbnail
                    if thumbnail_file_id and thumbnail_file_id != new_thumb_id:
                        print(f"DEBUG_LOG [process_review_job] Deleting old thumbnail {thumbnail_file_id}...")
                        if storage_type == "r2":
                            delete_r2_object(r2_client, r2_bucket, thumbnail_file_id)
                        else:
                            delete_file(service, thumbnail_file_id)
                        print("DEBUG_LOG [process_review_job] Old thumbnail deleted.")

                    updated_thumbnail_file_id = new_thumb_id

                # Upload trimmed video if it was created
                if new_video_created:
                    print(f"DEBUG_LOG [process_review_job] Uploading rendered video to folder {folder_id}...")
                    rendered_name = "video_template_rendered.mp4" if need_template_render else "video_trimmed.mp4"
                    if storage_type == "r2":
                        new_vid_id = f"{folder_id.rstrip('/')}/{rendered_name}"
                        upload_to_r2(r2_client, r2_bucket, new_vid_id, source_video_path, "video/mp4")
                    else:
                        new_vid_id = upload_file(service, source_video_path, rendered_name, folder_id, "video/mp4")
                    print(f"DEBUG_LOG [process_review_job] Trimmed video uploaded. New ID: {new_vid_id}")
                    
                    # Delete old video
                    if video_file_id and video_file_id != new_vid_id:
                        print(f"DEBUG_LOG [process_review_job] Deleting old video {video_file_id}...")
                        if storage_type == "r2":
                            delete_r2_object(r2_client, r2_bucket, video_file_id)
                        else:
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

