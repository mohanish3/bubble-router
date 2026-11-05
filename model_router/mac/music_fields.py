"""Validated music controls shared by the studio and Router."""
import math
from pathlib import Path

MUSIC_MODEL = 'acestep-v15-turbo'
MUSIC_MODELS = (MUSIC_MODEL, 'acestep-v15-xl-turbo')


def music_payload(body, root=None):
    """Build a bounded ACE-Step request; source paths stay in approved folders."""
    def text(name, default='', maximum=500):
        value = body.get(name, default)
        if not isinstance(value, str) or len(value) > maximum:
            raise ValueError(f'{name} must be text of at most {maximum} characters.')
        return value.strip()

    def number(name, default, low, high, integer=False):
        value = body.get(name, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not low <= value <= high
                or (integer and not isinstance(value, int))):
            raise ValueError(f'{name} must be {"an integer" if integer else "a number"} from {low} to {high}.')
        return value

    prompt = text('prompt', maximum=5000)
    if not prompt:
        raise ValueError('Describe the music you want to make.')
    model = text('model', MUSIC_MODEL, 60)
    if model not in MUSIC_MODELS:
        raise ValueError('Choose an installed ACE-Step music model.')
    task = text('task_type', 'text2music', 20)
    if task not in ('text2music', 'cover'):
        raise ValueError('Choose new music or remix.')
    instrumental = body.get('instrumental', False)
    enhance = body.get('enhance_prompt', True)
    thinking = body.get('thinking', True)
    if not all(isinstance(value,bool) for value in (instrumental,enhance,thinking)):
        raise ValueError('instrumental, enhance_prompt and thinking must be true or false.')
    result = {'model': model, 'prompt': prompt, 'task_type': task,
              'genre': text('genre'), 'mood': text('mood'), 'instruments': text('instruments'),
              'lyrics': text('lyrics', maximum=4000), 'vocal_language': text('vocal_language', 'en', 20),
              'keyscale': text('keyscale', maximum=40), 'timesignature': text('timesignature', '4', 1),
              'duration': number('duration', 60, 10, 300),
              'seed': number('seed', 42, 0, 2**53-1, True),
              'steps': number('steps', 8, 1, 32, True),
              'audio_cover_strength': number('audio_cover_strength', .65, 0, 1),
              'instrumental': instrumental, 'enhance_prompt': enhance, 'thinking':thinking}
    if result['timesignature'] not in ('2', '3', '4', '6'):
        raise ValueError('Choose time signature 2/4, 3/4, 4/4 or 6/8.')
    if result['vocal_language'] not in ('en','hi','zh','ja','ko','de','fr','es','it','pt','ru','unknown'):
        raise ValueError('Unsupported vocal language.')
    if body.get('bpm') is not None:
        result['bpm'] = number('bpm', 120, 30, 300, True)
    if body.get('caption') is not None:
        result['caption'] = text('caption', maximum=1500)
    source = text('source_audio', maximum=2000)
    if task == 'cover' and not source:
        raise ValueError('Upload audio or enter its local path before remixing.')
    if source:
        if task != 'cover':
            raise ValueError('Reference audio requires remix mode.')
        if root is not None:
            try:
                path = Path(source).expanduser().resolve(strict=True)
            except (OSError, RuntimeError):
                raise ValueError('Source audio file does not exist.') from None
            roots = [Path(root)/p for p in ('outputs','test-assets','state/uploads')]
            if (not any(path.is_relative_to(p.resolve()) for p in roots)
                    or not path.is_file() or path.stat().st_size > 100*1024**2):
                raise ValueError('Use audio under AI outputs, test-assets or uploads (up to 100 MB).')
            source = str(path)
        result['source_audio'] = source
    return result


def music_caption(payload):
    """Explicit controls remain authoritative after prompt enhancement."""
    parts = [payload.get('caption') or payload['prompt']]
    for name in ('genre','mood','instruments'):
        if payload.get(name):
            parts.append(f'{name}: {payload[name]}')
    if payload.get('instrumental'):
        parts.append('Instrumental, no vocals')
    return '. '.join(parts)
