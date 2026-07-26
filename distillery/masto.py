"""Post a video to Mastodon via mastodon.py."""
import logging
import time

from . import config

log = logging.getLogger("distillery")


def video_size_limit(instance_url, access_token):
    """Ask the instance its own video size cap, rather than guessing one."""
    try:
        from mastodon import Mastodon
        m = Mastodon(access_token=access_token, api_base_url=instance_url)
        conf = (m.instance() or {}).get("configuration") or {}
        lim = ((conf.get("media_attachments") or {}).get("video_size_limit"))
        return int(lim) if lim else None
    except Exception:                       # noqa: BLE001 - advisory only
        return None


def post_video(mp4_path, text, alt, secrets=None):
    from mastodon import Mastodon

    secrets = secrets or config.secrets()
    instance_url = (secrets.get("MASTODON_INSTANCE_URL") or "").strip()
    access_token = (secrets.get("MASTODON_ACCESS_TOKEN") or "").strip()
    if not instance_url or not access_token:
        raise ValueError(
            f"MASTODON_INSTANCE_URL / MASTODON_ACCESS_TOKEN not set in "
            f"{config.SECRETS_FILE}")

    size = mp4_path.stat().st_size
    limit = video_size_limit(instance_url, access_token)
    if limit and size > limit:
        raise ValueError(
            f"video is {size / 1e6:.1f} MB but {instance_url} accepts at most "
            f"{limit / 1e6:.1f} MB — raise VIDEO_CRF or shorten the piece")

    log.info("Posting to Mastodon at %s (%.1f MB) ...", instance_url, size / 1e6)
    masto = Mastodon(access_token=access_token, api_base_url=instance_url)
    with open(mp4_path, "rb") as f:
        media = masto.media_post(f, mime_type="video/mp4", description=alt)
    # video is transcoded asynchronously; attaching before it has a URL 422s
    for _ in range(90):
        if media.get("url"):
            break
        time.sleep(5)
        media = masto.media(media["id"])
    masto.status_post(status=text, media_ids=[media], visibility="public")
    log.info("Posted to Mastodon (%s).", instance_url)
    return True
