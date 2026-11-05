"""Reference-image sizing shared by video clients and the H3 backend."""
import json
import shutil
import subprocess


def fit_reference_size(width, height):
    """Trim centered edges to multiples of 32 without resampling pixels.

    Each dimension loses at most 31 pixels; no input pixel cap is applied.
    """
    if type(width) is not int or type(height) is not int or min(width, height) < 2:
        raise ValueError('Starting image must be at least 2 pixels on each side')
    output_width = width // 32 * 32
    output_height = height // 32 * 32
    if min(output_width, output_height) < 64:
        raise ValueError('Starting image must retain at least 64 pixels per side after cropping')
    return output_width, output_height


def reference_video_size(path, target_width=None, target_height=None):
    """Read the displayed image size using the already-required FFprobe."""
    ffprobe = shutil.which('ffprobe') or '/opt/homebrew/bin/ffprobe'
    try:
        result = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height:stream_side_data=rotation',
            '-of', 'json', str(path)], capture_output=True, timeout=15, check=True)
        stream = json.loads(result.stdout)['streams'][0]
        width, height = int(stream['width']), int(stream['height'])
        for side_data in stream.get('side_data_list', []):
            if abs(float(side_data.get('rotation', 0))) % 180 == 90:
                width, height = height, width
                break
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise ValueError('Starting image could not be decoded for video sizing') from exc

    if target_width is None and target_height is None:
        return fit_reference_size(width, height)
    if (type(target_width) is not int or type(target_height) is not int
            or min(target_width, target_height) < 64
            or target_width % 32 or target_height % 32):
        raise ValueError('Video crop dimensions must be multiples of 32 and at least 64')
    # The backend scales the image to cover this canvas, so any decodable size fits.
    return target_width, target_height
