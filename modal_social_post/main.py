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
import base64
import re
import urllib.parse
import httpx

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

def _parse_json_value(value, default):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return value if value is not None else default

def _ensure_slash(value: str) -> str:
    value = (value or "").replace("\\", "/").strip()
    return value if value.endswith("/") else f"{value}/"

def _content_type_ext(content_type: str) -> str:
    content_type = (content_type or "").lower()
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    return ".jpg"

def _put_bytes_to_r2(client, bucket: str, key: str, data: bytes, content_type: str):
    with_retry(lambda: client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type))

def _download_bytes(url: str, headers: dict | None = None, timeout: int = 120) -> tuple[bytes, str]:
    def run():
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, headers=headers or {})
            response.raise_for_status()
            return response.content, response.headers.get("content-type", "image/jpeg")
    return with_retry(run)

def _scene_prompts(topic: str, caption: str | None, count: int) -> list[str]:
    anchor = caption.strip() if caption else topic
    beats = [
        "opening hook, bold composition, clear subject",
        "real-world context, useful detail, natural light",
        "close-up evidence, premium editorial look",
        "human moment, aspirational but realistic scene",
        "final takeaway, confident visual ending"
    ]
    return [
        f"{topic}. {beats[index % len(beats)]}. Context: {anchor}. Vertical 9:16 social video frame, "
        "photorealistic, high detail, no readable text, no logos, no watermark."
        for index in range(count)
    ]

def _fetch_pexels_images(topic: str, api_key: str | None, count: int) -> list[dict]:
    if not api_key:
        raise RuntimeError("Pexels key is not configured.")
    query = urllib.parse.quote((topic or "social post")[:100])
    url = f"https://api.pexels.com/v1/search?query={query}&per_page={max(count, 5)}&orientation=portrait"
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        response = client.get(url, headers={"Authorization": api_key})
        response.raise_for_status()
        photos = response.json().get("photos", [])

    images = []
    for idx, photo in enumerate(photos[:count]):
        src = photo.get("src") or {}
        image_url = src.get("portrait") or src.get("large2x") or src.get("original") or src.get("large")
        if not image_url:
            continue
        data, content_type = _download_bytes(image_url)
        images.append({
            "index": idx + 1,
            "bytes": data,
            "content_type": content_type,
            "provider": "Pexels",
            "model": "pexels",
            "description": f"Pexels stock image for: {topic}"
        })
    if len(images) < count:
        raise RuntimeError(f"Pexels returned {len(images)} usable images; {count} required.")
    return images

def _generate_openai_image(prompt: str, provider: dict) -> dict:
    api_key = provider.get("api_key")
    if not api_key:
        raise RuntimeError("OpenAI image key is missing.")
    model = provider.get("model") or "dall-e-3"
    body = {"model": model, "prompt": prompt, "n": 1, "size": "1024x1792", "quality": "standard"}
    with httpx.Client(timeout=180, follow_redirects=True) as client:
        response = client.post("https://api.openai.com/v1/images/generations",
                               headers={"Authorization": f"Bearer {api_key}"}, json=body)
        response.raise_for_status()
        data = response.json().get("data", [{}])[0]
    if data.get("b64_json"):
        return {"bytes": base64.b64decode(data["b64_json"]), "content_type": "image/png", "provider": "OpenAI", "model": model}
    if data.get("url"):
        image_bytes, content_type = _download_bytes(data["url"])
        return {"bytes": image_bytes, "content_type": content_type, "provider": "OpenAI", "model": model}
    raise RuntimeError("OpenAI response did not include an image.")

def _generate_gemini_image(prompt: str, provider: dict) -> dict:
    api_key = provider.get("api_key")
    if not api_key:
        raise RuntimeError("Gemini image key is missing.")
    model = provider.get("model") or "gemini-2.0-flash-exp"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(model)}:generateContent?key={urllib.parse.quote(api_key)}"
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}}
    with httpx.Client(timeout=180, follow_redirects=True) as client:
        response = client.post(url, json=body)
        response.raise_for_status()
        parts = (((response.json().get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            return {
                "bytes": base64.b64decode(inline["data"]),
                "content_type": inline.get("mimeType") or inline.get("mime_type") or "image/png",
                "provider": "Gemini",
                "model": model
            }
    raise RuntimeError("Gemini response did not include inline image data.")

def _generate_together_image(prompt: str, provider: dict) -> dict:
    api_key = provider.get("api_key")
    if not api_key:
        raise RuntimeError("TogetherAI image key is missing.")
    model = provider.get("model") or "black-forest-labs/FLUX.1-schnell"
    body = {"model": model, "prompt": prompt, "n": 1, "width": 1024, "height": 1792}
    with httpx.Client(timeout=180, follow_redirects=True) as client:
        response = client.post("https://api.together.xyz/v1/images/generations",
                               headers={"Authorization": f"Bearer {api_key}"}, json=body)
        response.raise_for_status()
        data = response.json().get("data", [{}])[0]
    if data.get("b64_json"):
        return {"bytes": base64.b64decode(data["b64_json"]), "content_type": "image/png", "provider": "TogetherAI", "model": model}
    if data.get("url"):
        image_bytes, content_type = _download_bytes(data["url"])
        return {"bytes": image_bytes, "content_type": content_type, "provider": "TogetherAI", "model": model}
    raise RuntimeError("TogetherAI response did not include an image.")

def _generate_stability_image(prompt: str, provider: dict) -> dict:
    api_key = provider.get("api_key")
    if not api_key:
        raise RuntimeError("Stability AI image key is missing.")
    model = provider.get("model") or "sd3"
    with httpx.Client(timeout=180, follow_redirects=True) as client:
        response = client.post(
            "https://api.stability.ai/v2beta/stable-image/generate/sd3",
            headers={"Authorization": f"Bearer {api_key}", "Accept": "image/*"},
            data={"prompt": prompt, "model": model, "output_format": "png"}
        )
        response.raise_for_status()
        return {"bytes": response.content, "content_type": response.headers.get("content-type", "image/png"), "provider": "StabilityAI", "model": model}

def _generate_pollinations_image(prompt: str) -> dict:
    url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?width=1080&height=1920&nologo=true"
    image_bytes, content_type = _download_bytes(url, timeout=180)
    return {"bytes": image_bytes, "content_type": content_type, "provider": "Pollinations", "model": "pollinations"}

def _generate_ai_images(topic: str, caption: str | None, providers: list[dict], count: int) -> list[dict]:
    provider_list = providers or [{"provider": "Pollinations", "api_key": "", "model": "pollinations"}]
    if not any(str(p.get("provider", "")).lower() == "pollinations" for p in provider_list):
        provider_list.append({"provider": "Pollinations", "api_key": "", "model": "pollinations"})

    images = []
    for index, prompt in enumerate(_scene_prompts(topic, caption, count), start=1):
        last_error = None
        for provider in provider_list:
            name = str(provider.get("provider", "")).strip().lower()
            try:
                if name == "openai":
                    image = _generate_openai_image(prompt, provider)
                elif name == "gemini":
                    image = _generate_gemini_image(prompt, provider)
                elif name in ("together", "togetherai"):
                    image = _generate_together_image(prompt, provider)
                elif name in ("stability", "stabilityai"):
                    image = _generate_stability_image(prompt, provider)
                elif name == "pollinations":
                    image = _generate_pollinations_image(prompt)
                else:
                    continue
                image["index"] = index
                image["description"] = prompt
                images.append(image)
                break
            except Exception as exc:
                last_error = exc
                print(f"DEBUG_LOG [generate_post_asset] Image provider {provider.get('provider')} failed for scene {index}: {exc}")
        if len(images) < index:
            raise RuntimeError(f"All image providers failed for scene {index}: {last_error}")
    return images

def _save_ai_assets(client, bucket: str, asset_topic_folder: str | None, images: list[dict], job_id: str, topic: str) -> list[str]:
    if not asset_topic_folder:
        return []
    root = _ensure_slash(asset_topic_folder)
    saved = []
    stamp = time.strftime("%H%M%S")
    for image in images:
        index = int(image["index"])
        asset_folder = f"{root}ai-image-scene-{index:03d}-{stamp}/"
        ext = _content_type_ext(image.get("content_type", "image/jpeg"))
        media_key = f"{asset_folder}scene-{index:03d}{ext}"
        description = image.get("description") or topic
        metadata = {
            "kind": "ai-image",
            "sceneIndex": index,
            "durationSeconds": image.get("duration_seconds"),
            "provider": image.get("provider"),
            "model": image.get("model"),
            "jobId": job_id,
            "topic": topic,
            "savedAtUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        _put_bytes_to_r2(client, bucket, media_key, image["bytes"], image.get("content_type", "image/jpeg"))
        _put_bytes_to_r2(client, bucket, f"{asset_folder}description.txt", description.encode("utf-8"), "text/plain")
        _put_bytes_to_r2(client, bucket, f"{asset_folder}metadata.json", json.dumps(metadata).encode("utf-8"), "application/json")
        saved.append(media_key)
    return saved

def _image_unit_cost(provider: str) -> float:
    provider_key = (provider or "").strip().lower()
    if provider_key in ("pexels", "pixabay", "pollinations"):
        return 0.0
    if provider_key in ("together", "togetherai"):
        return 0.0019
    if provider_key == "gemini":
        return 0.02
    if provider_key == "openai":
        return 0.04
    if provider_key in ("stability", "stabilityai"):
        return 0.01
    return 0.04

def _build_cost_summary(images: list[dict], use_stock: bool) -> list[dict]:
    if use_stock:
        return []

    grouped: dict[tuple[str, str], int] = {}
    for image in images:
        provider = str(image.get("provider") or "").strip()
        model = str(image.get("model") or provider or "default").strip()
        if not provider:
            continue
        key = (provider, model)
        grouped[key] = grouped.get(key, 0) + 1

    summary: list[dict] = []
    for (provider, model), quantity in grouped.items():
        rate = _image_unit_cost(provider)
        cost = round(rate * quantity, 6)
        if cost <= 0:
            continue
        summary.append({
            "category": "ImageGen",
            "provider": provider,
            "model": model,
            "quantity": quantity,
            "unit": "images",
            "costUsd": cost
        })
    return summary

def _render_one_minute_video(tmpdir: str, images: list[dict], duration_seconds: int, image_duration: int) -> str:
    encoder = detect_encoder()
    codec, extra_args = ("h264_nvenc", ["-preset", "p4", "-rc", "vbr", "-cq", "23"]) if encoder == "h264_nvenc" else ("libx264", ["-preset", "fast", "-crf", "23"])
    clips = []
    for image in images:
        index = int(image["index"])
        ext = _content_type_ext(image.get("content_type", "image/jpeg"))
        image_path = os.path.join(tmpdir, f"scene_{index:03d}{ext}").replace("\\", "/")
        with open(image_path, "wb") as handle:
            handle.write(image["bytes"])
        clip_path = os.path.join(tmpdir, f"clip_{index:03d}.mp4").replace("\\", "/")
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", image_path,
            "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-c:v", codec, *extra_args, "-t", str(image_duration), "-r", "30", "-an", clip_path
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=True)
        clips.append(clip_path)

    concat_path = os.path.join(tmpdir, "concat.txt").replace("\\", "/")
    with open(concat_path, "w", encoding="utf-8") as handle:
        for clip in clips:
            handle.write(f"file '{clip}'\n")
    merged = os.path.join(tmpdir, "merged.mp4").replace("\\", "/")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_path, "-c", "copy", merged],
                   capture_output=True, text=True, timeout=300, check=True)
    final = os.path.join(tmpdir, "final.mp4").replace("\\", "/")
    subprocess.run(["ffmpeg", "-y", "-i", merged, "-t", str(duration_seconds), "-c:v", "copy", "-an", final],
                   capture_output=True, text=True, timeout=120, check=True)
    return final

def generate_post_asset_impl(payload: dict):
    """Source/generate post media, save optional AI assets, render/upload final post asset, callback."""
    callback_url = payload["callback_url"]
    callback_secret = payload["callback_secret"]
    r2_config = _parse_json_value(payload.get("r2_config"), {})
    settings = _parse_json_value(payload.get("settings"), {})
    client = get_r2_client(r2_config)
    bucket = r2_config.get("bucket_name", "")
    job_id = str(payload.get("job_id"))

    try:
        print(f"DEBUG_LOG [generate_post_asset] Starting job {job_id}", flush=True)
        post_type = settings.get("post_type", "PhotoCarousel")
        topic = settings.get("topic") or "Untitled social post"
        caption = settings.get("caption") or ""
        use_stock = bool(settings.get("use_stock"))
        duration_seconds = min(int(settings.get("duration_seconds") or 60), 60)
        image_duration = max(1, min(int(settings.get("image_duration_seconds") or 12), duration_seconds))
        output_folder = _ensure_slash(settings.get("output_folder") or payload.get("output_folder") or "")
        media_source = "StockSearch" if use_stock else "AiGenerated"
        required_images = 5 if post_type in ("ShortVideo", "SocialShort") else 1

        if use_stock:
            print(f"DEBUG_LOG [generate_post_asset] Fetching {required_images} Pexels images for job {job_id}", flush=True)
            images = _fetch_pexels_images(topic, settings.get("pexels_api_key"), required_images)
        else:
            print(f"DEBUG_LOG [generate_post_asset] Generating {required_images} AI images for job {job_id}", flush=True)
            images = _generate_ai_images(topic, caption, settings.get("image_providers") or [], required_images)

        asset_keys = _save_ai_assets(client, bucket, settings.get("asset_library_topic_folder"), images, job_id, topic) if not use_stock else []
        description_key = f"{output_folder}description.txt"
        _put_bytes_to_r2(client, bucket, description_key, (caption or topic).encode("utf-8"), "text/plain")

        outputs = {
            "MediaSource": media_source,
            "ImageCount": str(len(images)),
            "DurationSeconds": str(duration_seconds),
            "DescriptionKey": description_key,
            "AssetKeysJson": json.dumps(asset_keys)
        }
        cost_summary = _build_cost_summary(images, use_stock)

        with tempfile.TemporaryDirectory() as tmpdir:
            if post_type in ("ShortVideo", "SocialShort"):
                final = _render_one_minute_video(tmpdir, images[:5], duration_seconds, image_duration)
                output_key = f"{output_folder}generated_image_video_{job_id}.mp4"
                upload_to_r2(client, bucket, output_key, final, "video/mp4")
                outputs["RenderedPostKey"] = output_key
            else:
                image = images[0]
                ext = _content_type_ext(image.get("content_type", "image/jpeg"))
                output_key = f"{output_folder}generated_post_image_{job_id}{ext}"
                _put_bytes_to_r2(client, bucket, output_key, image["bytes"], image.get("content_type", "image/jpeg"))
                outputs["ImageKey"] = output_key

        send_callback(callback_url, callback_secret, {
            "success": True,
            "outputFilesJson": json.dumps(outputs),
            "costSummaryJson": json.dumps(cost_summary)
        })
        print(f"DEBUG_LOG [generate_post_asset] Completed job {job_id}", flush=True)
    except Exception as e:
        import traceback
        print("DEBUG_LOG [generate_post_asset] EXCEPTION OCCURRED:")
        traceback.print_exc()
        send_callback(callback_url, callback_secret, {"success": False, "errorMessage": str(e)})

@app.function(image=nvidia_image, gpu="A10G", timeout=1200, memory=8192)
def generate_post_asset(payload: dict):
    return generate_post_asset_impl(payload)

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
        "GeneratePostAsset": generate_post_asset,
    }
    fn = function_map.get(step_name)
    if fn is None:
        raise fastapi.HTTPException(status_code=400, detail=f"Unknown step: {step_name}")
    fn.spawn(body)
    return {"dispatched": True}

