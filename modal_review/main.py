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

def generate_text_description(prompt: str, provider: str, model: str, api_key: str) -> str:
    provider_key = (provider or "Gemini").strip().lower()
    resolved_model = (model or "").strip()

    if provider_key == "gemini":
        model_name = resolved_model or "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent?key={api_key}"
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

    if provider_key == "deepseek":
        url = "https://api.deepseek.com/chat/completions"
        resolved_model = resolved_model or "deepseek-chat"
    elif provider_key == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        resolved_model = resolved_model or "openai/gpt-4o-mini"
    elif provider_key == "alibaba":
        url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
        resolved_model = resolved_model or "qwen-plus"
    else:
        url = "https://api.openai.com/v1/chat/completions"
        resolved_model = resolved_model or "gpt-4o-mini"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    if provider_key == "openrouter":
        headers["HTTP-Referer"] = "https://heyjivu.local"
        headers["X-Title"] = "HeyJivu"

    body = {
        "model": resolved_model,
        "messages": [
            {
                "role": "system",
                "content": "You write concise publishing-ready video descriptions."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    response = httpx.post(url, json=body, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise ValueError(f"{provider} API returned invalid response: {data}")

def send_callback(callback_url: str, payload: dict, callback_secret: str = None):
    if not callback_url:
        return
    headers = {}
    if callback_secret:
        headers["X-Modal-Secret"] = callback_secret
    print(f"Sending callback to {callback_url} with payload keys: {list(payload.keys())} and headers: {list(headers.keys())}")
    def _post():
        response = httpx.post(callback_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response

    with_retry(lambda: _post())

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
        "effectId", "effectName", "audioConflictStrategy", "soundtrackId",
        "soundtrackName", "soundtrackPreviewUrl", "previewUrl", "audioUrl",
        "soundtrackUrl", "soundtrackStartTime", "soundtrackEndTime",
        "audioStartTime", "audioEndTime", "soundtrackVolume", "volume"
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

def _srt_timestamp(seconds: float) -> str:
    value = max(0.0, float(seconds or 0))
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    secs = int(value % 60)
    millis = int(round((value - int(value)) * 1000))
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def _caption_text_for_video(title: str | None, description: str | None) -> str:
    text = " ".join(part.strip() for part in [title or "", description or ""] if part and part.strip())
    if not text:
        text = "Review this clip and add a strong opening caption."
    first_sentence = text.split(". ")[0].strip()
    return first_sentence[:160]

def _caption_config_from_edit(template_edit: dict) -> dict:
    config = _flatten_template_config(template_edit)
    caption = _first_value(config.get("caption"), config.get("captions"), config.get("captionSettings"), config.get("caption_settings"))
    return _as_dict(caption)

def _ass_color(value, fallback="#ffffff"):
    raw = str(value or fallback).strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in raw):
        raw = str(fallback).strip().lstrip("#")
    red, green, blue = raw[0:2], raw[2:4], raw[4:6]
    return f"&H00{blue}{green}{red}"

def _caption_style(config: dict) -> str:
    color_mode = str(config.get("colorMode") or config.get("color_mode") or "white").lower()
    style = str(config.get("style") or "bold").lower()
    position = str(config.get("position") or "bottom").lower()
    try:
        x_pct = float(config.get("xPercent") or config.get("x_percent") or 50)
        y_pct = float(config.get("yPercent") or config.get("y_percent") or 82)
    except Exception:
        x_pct, y_pct = 50.0, 82.0

    if color_mode == "yellow":
        primary = _ass_color("#ffe86b")
    elif color_mode == "cyan":
        primary = _ass_color("#72f6ff")
    elif color_mode == "mixed":
        primary = _ass_color("#5fe9ff")
    else:
        primary = _ass_color("#ffffff")

    if position == "top":
        alignment, margin_v = 8, 70
    elif position == "center":
        alignment, margin_v = 5, 20
    else:
        horizontal = 1 if x_pct < 35 else 3 if x_pct > 65 else 2
        vertical = 7 if y_pct < 33 else 4 if y_pct < 66 else 1
        alignment = vertical + (horizontal - 1)
        margin_v = max(40, int(1080 * (y_pct / 100 if y_pct < 50 else (100 - y_pct) / 100)))

    border_style = 3 if style == "boxed" else 1
    outline = 3 if style in ("bold", "karaoke", "boxed") else 2
    back = _ass_color("#111111") if style == "boxed" else "&H8A000000"
    if style == "karaoke":
        primary = _ass_color("#ffdd57" if color_mode in ("white", "yellow") else "#57e0ff")

    return ",".join([
        "FontName=Arial",
        "FontSize=22",
        f"PrimaryColour={primary}",
        "OutlineColour=&H00000000",
        f"BackColour={back}",
        f"BorderStyle={border_style}",
        f"Outline={outline}",
        "Shadow=1",
        f"Alignment={alignment}",
        f"MarginV={margin_v}",
    ])

def _write_basic_srt(path: str, title: str | None, description: str | None, duration: float, caption_config: dict | None = None):
    caption = _caption_text_for_video(title, description)
    config = caption_config or {}
    if str(config.get("language") or "").lower() == "ur" and not caption.strip():
        caption = "یہاں کیپشن دکھائی دیں گے"
    end = max(2.0, min(duration or 6.0, 6.0))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"1\n00:00:00,000 --> {_srt_timestamp(end)}\n{caption}\n")

def _burn_basic_captions(input_path: str, output_path: str, srt_path: str, caption_config: dict | None = None):
    escaped_srt = srt_path.replace("\\", "/").replace(":", "\\:")
    force_style = _caption_style(caption_config or {})
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"subtitles='{escaped_srt}':force_style='{force_style}'",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg caption burn failed:\n{result.stderr}")

def _resolve_external_audio(config: dict, tmpdir: str, r2_client=None, r2_bucket: str = "", suffix: str = ""):
    audio_url = _first_value(config.get("audioUrl"), config.get("audio_url"), config.get("previewUrl"), config.get("soundtrackPreviewUrl"), config.get("soundtrackUrl"))
    audio_key = _first_value(config.get("audioFileKey"), config.get("audio_file_key"), config.get("fileKey"), config.get("soundtrackFileKey"))
    if audio_key and r2_client and r2_bucket:
        path = os.path.join(tmpdir, f"template_audio{suffix}.mp3").replace("\\", "/")
        download_from_r2(r2_client, r2_bucket, audio_key, path)
        return path
    if audio_url and str(audio_url).startswith(("http://", "https://")):
        path = os.path.join(tmpdir, f"template_audio{suffix}.mp3").replace("\\", "/")
        with httpx.stream("GET", str(audio_url), timeout=60, follow_redirects=True) as response:
            response.raise_for_status()
            with open(path, "wb") as fh:
                for chunk in response.iter_bytes():
                    if chunk:
                        fh.write(chunk)
        return path
    return None

def _soundtrack_entries(config: dict):
    soundtracks = config.get("soundtracks")
    if isinstance(soundtracks, list):
        return [item for item in soundtracks if isinstance(item, dict)]
    if _first_value(config.get("audioUrl"), config.get("previewUrl"), config.get("soundtrackPreviewUrl"), config.get("soundtrackUrl"), config.get("audioFileKey"), config.get("soundtrackFileKey")):
        return [config]
    return []

def _resolve_external_audio_tracks(config: dict, tmpdir: str, r2_client=None, r2_bucket: str = ""):
    tracks = []
    for index, entry in enumerate(_soundtrack_entries(config), start=1):
        audio_path = _resolve_external_audio(entry, tmpdir, r2_client, r2_bucket, f"_{index}")
        if audio_path:
            tracks.append((entry, audio_path))
    return tracks

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
    audio_tracks = _resolve_external_audio_tracks(config, tmpdir, r2_client, r2_bucket)
    external_audio_path = audio_tracks[0][1] if audio_tracks else None
    vf = _build_video_filters(config, template_edit, asset_type, input_path, external_audio_path)

    cmd = ["ffmpeg", "-y", "-i", input_path]
    for _, audio_path in audio_tracks:
        if strategy == "repeataudio":
            cmd.extend(["-stream_loop", "-1"])
        cmd.extend(["-i", audio_path])

    has_original_audio = _has_audio_stream(input_path)
    filter_parts = [f"[0:v]{vf}[vout]"]
    map_args = ["-map", "[vout]"]

    if audio_tracks:
        duration_mode = "longest" if strategy == "extendvideo" else "first"
        mix_inputs = ["[0:a]"] if has_original_audio else []
        for track_index, (track_config, _) in enumerate(audio_tracks, start=1):
            volume = float(_first_value(track_config.get("soundtrackVolume"), track_config.get("volume"), config.get("soundtrackVolume"), config.get("volume"), 50) or 50)
            volume = volume / 100.0 if volume > 1 else volume
            start_time = max(0.0, float(_first_value(track_config.get("soundtrackStartTime"), track_config.get("audioStartTime"), track_config.get("startTime"), 0) or 0))
            end_value = _first_value(track_config.get("soundtrackEndTime"), track_config.get("audioEndTime"), track_config.get("endTime"))
            try:
                end_time = float(end_value) if end_value is not None else 0.0
            except Exception:
                end_time = 0.0
            label = f"music{track_index}"
            audio_chain = f"[{track_index}:a]atrim=start=0"
            if end_time > start_time:
                audio_chain += f":duration={end_time - start_time:.3f}"
            delay_ms = int(start_time * 1000)
            audio_chain += f",asetpts=PTS-STARTPTS,volume={volume:.3f}"
            if delay_ms > 0:
                audio_chain += f",adelay={delay_ms}|{delay_ms}"
            filter_parts.append(f"{audio_chain}[{label}]")
            mix_inputs.append(f"[{label}]")

        if len(mix_inputs) > 1:
            filter_parts.append(f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration={duration_mode}:dropout_transition=2[aout]")
            map_args.extend(["-map", "[aout]"])
        else:
            map_args.extend(["-map", mix_inputs[0] if mix_inputs else "0:a?"])
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
    if audio_tracks and strategy in ("trimaudio", "repeataudio"):
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

def create_drive_folder(service, parent_folder_id: str, folder_name: str) -> str:
    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id]
    }
    folder = service.files().create(body=metadata, fields="id").execute()
    return folder.get("id")

def revision_folder_name() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())

def archive_r2_object(client, bucket: str, key: str, folder_id: str):
    if not key:
        return
    try:
        basename = os.path.basename(str(key).rstrip("/")) or "asset"
        revision_key = f"{folder_id.rstrip('/')}/_revisions/{revision_folder_name()}/{basename}"
        client.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": key}, Key=revision_key)
        client.delete_object(Bucket=bucket, Key=key)
        print(f"DEBUG_LOG [archive_r2_object] Archived {key} to {revision_key}")
    except Exception as e:
        print(f"WARNING: Failed to archive R2 object {key}: {e}")

def archive_drive_file(service, file_id: str, folder_id: str):
    if not file_id:
        return
    try:
        revisions_root_id = create_drive_folder(service, folder_id, "_revisions")
        revision_id = create_drive_folder(service, revisions_root_id, revision_folder_name())
        move_file_or_folder(service, file_id, revision_id)
        print(f"DEBUG_LOG [archive_drive_file] Archived file {file_id} under _revisions.")
    except Exception as e:
        print(f"WARNING: Failed to archive Drive file {file_id}: {e}")

def archive_storage_object(storage_type: str, service, r2_client, r2_bucket: str, file_id: str, folder_id: str):
    if not file_id:
        return
    if storage_type == "r2":
        archive_r2_object(r2_client, r2_bucket, file_id, folder_id)
    else:
        archive_drive_file(service, file_id, folder_id)

def replace_key_prefix(key: str | None, source_prefix: str | None, target_prefix: str | None):
    if not key or not source_prefix or not target_prefix:
        return key
    source = source_prefix.strip("/")
    target = target_prefix.strip("/")
    normalized = key.strip("/")
    if normalized == source:
        return target
    if normalized.startswith(source + "/"):
        return f"{target}/{normalized[len(source):].lstrip('/')}"
    return key

# ── Orchestrated Worker Function ──────────────────────────────────────────────

@app.function(timeout=600, memory=4096)
def process_review_job(payload: dict):
    video_id = payload.get("video_id")
    is_finalize = payload.get("is_finalize", False)
    storage_type = str(payload.get("storage_type") or "drive").lower()
    generate_video = payload.get("generate_video", False)
    generate_thumbnail = payload.get("generate_thumbnail", False)
    generate_description = payload.get("generate_description", False)
    generate_captions = payload.get("generate_captions", False)
    start_time = payload.get("start_time")
    end_time = payload.get("end_time")
    thumbnail_timestamp = payload.get("thumbnail_timestamp")
    try:
        thumbnail_count = int(payload.get("thumbnail_count") or 1)
    except Exception:
        thumbnail_count = 1
    thumbnail_count = max(1, min(8, thumbnail_count))
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
    text_ai_provider = payload.get("text_ai_provider") or "Gemini"
    text_ai_model = payload.get("text_ai_model") or ""
    text_ai_api_key = payload.get("text_ai_api_key") or gemini_api_key
    callback_url = payload.get("callback_url")
    callback_secret = payload.get("callback_secret")
    r2_client = None
    r2_bucket = ""

    print(f"DEBUG_LOG [process_review_job] Started for video_id: {video_id}. storage_type: {storage_type}, is_finalize: {is_finalize}, generate_video: {generate_video}, generate_thumbnail: {generate_thumbnail}, generate_description: {generate_description}, generate_captions: {generate_captions}, has_template_edit: {bool(template_edit)}")
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
                            if video_file_id == rendered_key:
                                archive_storage_object(storage_type, service, r2_client, r2_bucket, video_file_id, folder_id)
                            upload_to_r2(r2_client, r2_bucket, rendered_key, rendered_video_path, "video/mp4")
                            if video_file_id != rendered_key:
                                archive_storage_object(storage_type, service, r2_client, r2_bucket, video_file_id, folder_id)
                            video_file_id = rendered_key
                        else:
                            new_vid_id = upload_file(service, rendered_video_path, "video_template_rendered.mp4", folder_id, "video/mp4")
                            if video_file_id and video_file_id != new_vid_id:
                                archive_storage_object(storage_type, service, r2_client, r2_bucket, video_file_id, folder_id)
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
                "success": True,
                "sourceFolderId": folder_id,
                "targetFolderId": target_folder_id,
                "videoFileId": replace_key_prefix(video_file_id, folder_id, target_folder_id),
                "thumbnailFileId": replace_key_prefix(thumbnail_file_id, folder_id, target_folder_id)
            }, callback_secret)
            print(f"DEBUG_LOG [process_review_job] Finalize success callback sent.")
            return

        # ── Case B: Regenerate video parts, thumbnail or description ──────────
        updated_video_file_id = video_file_id
        updated_thumbnail_file_id = thumbnail_file_id
        updated_description = description

        # 1. Handle Video Trimming / Thumbnail Extraction (needs file download)
        need_template_render = bool(template_edit)
        need_download = (generate_video and start_time is not None and end_time is not None) or generate_thumbnail or need_template_render or generate_captions
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

                if generate_captions:
                    caption_srt_path = os.path.join(tmpdir, "review_captions.srt").replace("\\", "/")
                    captioned_video_path = os.path.join(tmpdir, "captioned.mp4").replace("\\", "/")
                    caption_config = _caption_config_from_edit(template_edit)
                    _write_basic_srt(caption_srt_path, title, description, _ffprobe_duration(source_video_path), caption_config)
                    print(f"DEBUG_LOG [process_review_job] Burning regenerated captions from {caption_srt_path} -> {captioned_video_path}...")
                    _burn_basic_captions(source_video_path, captioned_video_path, caption_srt_path, caption_config)
                    source_video_path = captioned_video_path
                    new_video_created = True

                # Handle Thumbnail Extraction
                if generate_thumbnail:
                    video_duration = _ffprobe_duration(source_video_path) or 0.0
                    if thumbnail_timestamp is not None:
                        base_time = max(0.0, float(thumbnail_timestamp))
                        candidate_times = [min(video_duration or base_time, base_time)]
                    else:
                        safe_duration = max(2.0, video_duration)
                        candidate_times = [
                            min(safe_duration - 0.2, max(0.2, ((i + 1) / (thumbnail_count + 1)) * safe_duration))
                            for i in range(thumbnail_count)
                        ]

                    new_thumb_id = None
                    for candidate_index, t_sec in enumerate(candidate_times, start=1):
                        thumb_name = "thumbnail.png" if candidate_index == 1 else f"thumbnail_{candidate_index}.png"
                        thumb_path = os.path.join(tmpdir, thumb_name)
                        print(f"DEBUG_LOG [process_review_job] Extracting thumbnail candidate {candidate_index} at {t_sec} seconds to {thumb_path}...")

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
                        if result.returncode != 0:
                            print(f"DEBUG_LOG [process_review_job] Thumbnail FAILED! stderr: {result.stderr}")
                            raise RuntimeError(f"FFmpeg thumbnail extraction failed:\n{result.stderr}")
                        print(f"DEBUG_LOG [process_review_job] Thumbnail candidate file size: {os.path.getsize(thumb_path)} bytes")

                        print(f"DEBUG_LOG [process_review_job] Uploading thumbnail candidate to folder {folder_id}...")
                        if storage_type == "r2":
                            candidate_thumb_id = f"{folder_id.rstrip('/')}/{thumb_name}"
                            if thumbnail_file_id == candidate_thumb_id:
                                archive_storage_object(storage_type, service, r2_client, r2_bucket, thumbnail_file_id, folder_id)
                            upload_to_r2(r2_client, r2_bucket, candidate_thumb_id, thumb_path, "image/png")
                        else:
                            candidate_thumb_id = upload_file(service, thumb_path, thumb_name, folder_id, "image/png")
                        if candidate_index == 1:
                            new_thumb_id = candidate_thumb_id
                        print(f"DEBUG_LOG [process_review_job] Thumbnail candidate uploaded. New ID: {candidate_thumb_id}")
                    
                    # Archive old thumbnail out of the active item folder.
                    if new_thumb_id and thumbnail_file_id and thumbnail_file_id != new_thumb_id:
                        print(f"DEBUG_LOG [process_review_job] Archiving old thumbnail {thumbnail_file_id}...")
                        archive_storage_object(storage_type, service, r2_client, r2_bucket, thumbnail_file_id, folder_id)
                        print("DEBUG_LOG [process_review_job] Old thumbnail archived.")

                    updated_thumbnail_file_id = new_thumb_id

                # Upload trimmed video if it was created
                if new_video_created:
                    print(f"DEBUG_LOG [process_review_job] Uploading rendered video to folder {folder_id}...")
                    rendered_name = "video_template_rendered.mp4" if need_template_render else "video_trimmed.mp4"
                    if storage_type == "r2":
                        new_vid_id = f"{folder_id.rstrip('/')}/{rendered_name}"
                        if video_file_id == new_vid_id:
                            archive_storage_object(storage_type, service, r2_client, r2_bucket, video_file_id, folder_id)
                        upload_to_r2(r2_client, r2_bucket, new_vid_id, source_video_path, "video/mp4")
                    else:
                        new_vid_id = upload_file(service, source_video_path, rendered_name, folder_id, "video/mp4")
                    print(f"DEBUG_LOG [process_review_job] Trimmed video uploaded. New ID: {new_vid_id}")
                    
                    # Archive old video out of the active item folder.
                    if video_file_id and video_file_id != new_vid_id:
                        print(f"DEBUG_LOG [process_review_job] Archiving old video {video_file_id}...")
                        archive_storage_object(storage_type, service, r2_client, r2_bucket, video_file_id, folder_id)
                        print("DEBUG_LOG [process_review_job] Old video archived.")

                    updated_video_file_id = new_vid_id

        # 2. Handle Description Generation
        if generate_description:
            if not text_ai_api_key:
                raise ValueError("Text AI API key is required to generate description.")
            
            prompt = (
                f"Generate a polished social media caption and video description for a video titled: '{title or ''}'. "
                f"Use this current reviewer context as source material, but do not copy it verbatim: '{description or ''}'. "
                "Write a fresh publishing-ready description, avoid raw transcript style, keep it engaging and concise, "
                "and return ONLY the description without metadata or title."
            )
            print(f"DEBUG_LOG [process_review_job] Generating description using {text_ai_provider} ({text_ai_model or 'default model'})...")
            updated_description = generate_text_description(prompt, text_ai_provider, text_ai_model, text_ai_api_key)
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

