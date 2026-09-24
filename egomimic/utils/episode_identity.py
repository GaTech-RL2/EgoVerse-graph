"""Local episode identity conversions without SQL or cloud imports."""

from datetime import datetime, timezone


def timestamp_ms_to_episode_hash(timestamp_ms):
    """Convert UTC epoch milliseconds to YYYY-MM-DD-HH-MM-SS-ffffff."""
    seconds, milliseconds = divmod(int(timestamp_ms), 1000)
    value = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(
        microsecond=milliseconds * 1000
    )
    return value.strftime("%Y-%m-%d-%H-%M-%S-%f")
