import subprocess
import boto3
from botocore.config import Config
from modal_common.common import with_retry


def get_r2_client(r2_config: dict):
    if not r2_config:
        return None
    return boto3.client(
        service_name="s3",
        endpoint_url=r2_config.get("endpoint"),
        aws_access_key_id=r2_config.get("access_key_id"),
        aws_secret_access_key=r2_config.get("secret_access_key"),
        config=Config(signature_version="s3v4")
    )


def download_from_r2(client, bucket: str, key: str, dest_path: str):
    with_retry(lambda: client.download_file(bucket, key, dest_path))


def upload_to_r2(client, bucket: str, key: str, src_path: str, content_type: str = "video/mp4"):
    with_retry(lambda: client.upload_file(src_path, bucket, key, ExtraArgs={"ContentType": content_type}))


def detect_encoder() -> str:
    try:
        result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10, check=True)
        if "h264_nvenc" in result.stdout:
            return "h264_nvenc"
    except Exception:
        pass
    return "libx264"
