"""Real image/video processing with a tiny substitute for the H3 model call."""
import json
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace
import zlib

import pytest

from model_router.mac.media import MediaBackend
from model_router.mac.video_sizing import fit_reference_size, reference_video_size


def image(width, height):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    # Top and bottom stripes detect any accidental vertical cropping.
    green = b'\x00' + b''.join(bytes((x % 256, 255, 0)) for x in range(width))
    red = b'\x00' + b'\xff\x00\x00' * width
    blue = b'\x00' + b'\x00\x00\xff' * width
    rows = red * 60 + green * (height - 120) + blue * 60
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))


@pytest.mark.parametrize('source, expected', [
    ((800, 1200), (800, 1184)), ((1200, 800), (1184, 800)),
    ((805, 1213), (800, 1184)), ((1345, 769), (1344, 768)),
    ((512, 512), (512, 512)), ((320, 240), (320, 224)),
    ((128, 96), (128, 96)), ((511, 511), (480, 480)),
    ((10000, 100), (9984, 96)),
])
def test_fit_only_trims_edges_without_scaling(source, expected):
    width, height = fit_reference_size(*source)
    assert (width, height) == expected
    assert width % 32 == height % 32 == 0
    assert 0 <= source[0] - width < 32
    assert 0 <= source[1] - height < 32
    assert fit_reference_size(width, height) == (width, height)


@pytest.mark.parametrize('source', [(0, 100), (100, 0), (40, 40), (True, 128), (128.0, 128)])
def test_impossible_inputs_fail_without_upscaling_or_stretching(source):
    with pytest.raises(ValueError):
        fit_reference_size(*source)


def test_large_first_frame_is_accepted_at_selected_video_size(tmp_path):
    assets = tmp_path / 'test-assets'
    assets.mkdir()
    photo = assets / 'large.png'
    photo.write_bytes(image(1200, 1800))
    assert reference_video_size(photo) == (1184, 1792)
    backend = MediaBackend(None, {}, tmp_path)
    for field in ['first_frame', 'initial_image']:
        options = backend._video_options({'prompt': 'A camera pan', field: str(photo),
            'width': 704, 'height': 384, 'upscale': 4})
        assert options[:2] == (704, 384)
        assert options[-1] == 1
    with pytest.raises(ValueError, match='multiples of 32'):
        backend._video_options({'prompt': 'A camera pan', 'first_frame': str(photo),
            'width': 705, 'height': 384})
    with pytest.raises(ValueError, match='must not exceed'):
        backend._video_options({'prompt': 'A camera pan', 'first_frame': str(photo),
            'width': 1344, 'height': 1344})
    # Smaller sources are scaled up to cover the canvas instead of being rejected.
    assert reference_video_size(photo, 1280, 768) == (1280, 768)


def test_text_only_and_last_frame_only_keep_requested_dimensions(tmp_path):
    assets = tmp_path / 'test-assets'
    assets.mkdir()
    photo = assets / 'photo.png'
    photo.write_bytes(image(805, 1213))
    backend = MediaBackend(None, {}, tmp_path)
    for extra in [{}, {'last_frame': str(photo)}]:
        options = backend._video_options({'prompt': 'A camera pan', 'width': 1344,
                                         'height': 768, 'upscale': 2, **extra})
        assert options[:2] == (1344, 768)
        assert options[-1] == 2


@pytest.mark.asyncio
async def test_real_normalization_scales_to_cover_and_export_keeps_native_audio(tmp_path):
    assets = tmp_path / 'test-assets'
    assets.mkdir()
    photo = assets / 'photo.png'
    photo.write_bytes(image(805, 1213))
    original = photo.read_bytes()
    model = tmp_path / 'models/h3'
    model.mkdir(parents=True)
    h3 = tmp_path / 'repos/h3.c/h3'
    h3.parent.mkdir(parents=True)
    h3.touch()

    class Manager:
        calls = []
        prepared_size = None
        prepared_rows = None

        async def run_owned(self, name, args, *, cwd=None, env=None):
            self.calls.append(list(args))
            if args[0] == str(h3):
                first = args[args.index('--first-frame') + 1]
                self.prepared_path = Path(first)
                self.prepared_size = reference_video_size(first)
                raw = subprocess.run(['/opt/homebrew/bin/ffmpeg', '-v', 'error', '-i', first,
                    '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], capture_output=True, check=True).stdout
                self.prepared_rows = (raw[:3], raw[-3:])
                self.prepared_pixels = raw
                last = args[args.index('--last-frame') + 1]
                self.last_pixels = subprocess.run(['/opt/homebrew/bin/ffmpeg', '-v', 'error', '-i', last,
                    '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], capture_output=True, check=True).stdout
                # Stand in for model inference; all other processing runs for real.
                args = ['/opt/homebrew/bin/ffmpeg', '-y', '-v', 'error', '-loop', '1', '-i', first,
                    '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=32000', '-r', '24',
                    '-frames:v', args[args.index('--frames') + 1], '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', args[args.index('-o') + 1]]
            return subprocess.run(args, cwd=cwd, capture_output=True, timeout=30)

    manager = Manager()
    backend = MediaBackend(manager, {'minimax-h3': {'path': str(model)}}, tmp_path)
    result = await backend.generate_video({'prompt': 'A camera pan', 'first_frame': str(photo),
        'width': 800, 'height': 1184, 'duration': .25, 'upscale': 4, 'last_frame': str(photo)}, 'sizing-test')
    assert manager.prepared_size == (800, 1184)
    assert manager.prepared_rows == (b'\xff\x00\x00', b'\x00\x00\xff')
    assert len(manager.prepared_pixels) == 800 * 1184 * 3
    assert manager.last_pixels == manager.prepared_pixels
    assert not manager.prepared_path.exists()  # Ephemeral anchors are cleaned up.
    assert photo.read_bytes() == original
    assert (result['width'], result['height'], result['upscale']) == (800, 1184, None)
    assert result['actual_frames'] == 6
    probe = subprocess.run(['/opt/homebrew/bin/ffprobe', '-v', 'error', '-show_streams',
        '-of', 'json', result['artifact']], capture_output=True, check=True)
    assert {row['codec_type'] for row in json.loads(probe.stdout)['streams']} == {'audio', 'video'}
    prep = manager.calls[0]
    assert prep[prep.index('-vf') + 1] == ("scale=w='max(800\\,ceil(1184*iw/ih))':h='max(1184\\,ceil(800*ih/iw))'"
        ':flags=lanczos,crop=800:1184:floor((iw-ow)/2):floor((ih-oh)/2):exact=1,setsar=1')


@pytest.mark.asyncio
async def test_large_photo_is_downscaled_not_cropped_to_square_canvas(tmp_path):
    assets = tmp_path / 'test-assets'
    assets.mkdir()
    photo = assets / 'wide.png'
    photo.write_bytes(image(1600, 1200))
    model = tmp_path / 'models/h3'
    model.mkdir(parents=True)
    h3 = tmp_path / 'repos/h3.c/h3'
    h3.parent.mkdir(parents=True)
    h3.touch()

    class Manager:
        pixels = None

        async def run_owned(self, name, args, *, cwd=None, env=None):
            if args[0] == str(h3):
                first = args[args.index('--first-frame') + 1]
                self.size = reference_video_size(first)
                self.pixels = subprocess.run(['/opt/homebrew/bin/ffmpeg', '-v', 'error', '-i', first,
                    '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], capture_output=True, check=True).stdout
                args = ['/opt/homebrew/bin/ffmpeg', '-y', '-v', 'error', '-loop', '1', '-i', first, '-r', '24',
                    '-frames:v', args[args.index('--frames') + 1], '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', args[args.index('-o') + 1]]
            return subprocess.run(args, cwd=cwd, capture_output=True, timeout=30)

    manager = Manager()
    backend = MediaBackend(manager, {'minimax-h3': {'path': str(model)}}, tmp_path)
    await backend.generate_video({'prompt': 'A camera pan', 'first_frame': str(photo),
        'width': 512, 'height': 512, 'duration': .25}, 'cover-test')
    assert manager.size == (512, 512)
    # The full source height survives: the red top and blue bottom stripes are still in frame.
    assert manager.pixels[:3] == b'\xff\x00\x00'
    assert manager.pixels[-3:] == b'\x00\x00\xff'
