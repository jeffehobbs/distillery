"""Post a video to Bluesky via atproto."""
import logging
import os

from . import config, video

log = logging.getLogger("distillery")

# atproto's default httpx timeout (5 s) is far too short to upload tens of MB over a
# home connection; give the blob upload plenty of room.
UPLOAD_TIMEOUT_S = float(os.environ.get("DISTILLERY_BSKY_TIMEOUT", "600"))


def post_video(mp4_path, text, alt, secrets=None):
    from atproto import Client, models
    from atproto_client.request import Request

    secrets = secrets or config.secrets()
    handle = (secrets.get("BLUESKY_HANDLE") or "").strip()
    password = (secrets.get("BLUESKY_PASSWORD") or "").strip()
    if not handle or not password:
        raise ValueError(
            f"BLUESKY_HANDLE / BLUESKY_PASSWORD not set in {config.SECRETS_FILE} "
            f"— use an app password, not the account password.")
    log.info("Posting to Bluesky as %s (%.1f MB) ...", handle,
             mp4_path.stat().st_size / 1e6)
    client = Client(request=Request(timeout=UPLOAD_TIMEOUT_S))
    client.login(handle, password)
    with open(mp4_path, "rb") as f:
        data = f.read()
    client.send_video(
        text=text, video=data, video_alt=alt,
        video_aspect_ratio=models.AppBskyEmbedDefs.AspectRatio(
            width=config.VIDEO_WIDTH, height=config.VIDEO_HEIGHT))
    log.info("Posted to Bluesky (%s).", handle)
    return True
