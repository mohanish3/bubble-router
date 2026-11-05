"""Manager-owned ACE-Step music generation, with durable original/remix artifacts."""
import json
import shutil
import subprocess
from pathlib import Path
from .music_fields import music_payload


class MusicBackend:
    """Run one isolated music process under Router memory admission."""
    def __init__(self, manager, root):
        self.manager, self.root = manager, Path(root)

    def validate(self, body):
        payload = music_payload(body, self.root)
        if payload.get('source_audio'):
            probe = subprocess.run(['/opt/homebrew/bin/ffprobe','-v','error','-show_entries',
                'stream=codec_type:format=duration','-of','json',payload['source_audio']],
                capture_output=True,timeout=15)
            try:
                info = json.loads(probe.stdout)
                duration = float(info['format']['duration'])
                if probe.returncode or not 10 <= duration <= 300 or not any(s.get('codec_type')=='audio' for s in info['streams']):
                    raise ValueError('invalid audio')
            except (ValueError, KeyError):
                raise ValueError('Source must be a valid audio recording, 10–300 seconds long.') from None
            payload['duration'] = duration
        return payload

    async def generate(self, payload, job_id):
        paths = json.loads((self.root/'config/model-paths.json').read_text())
        entry = paths.get('ACE-Step/Ace-Step1.5', {})
        project = Path(entry.get('project',''))
        python = project/'.venv/bin/python'
        worker = self.root/'bin/music-worker.py'
        if not python.is_file() or not worker.is_file():
            raise RuntimeError('ACE-Step music is not installed.')
        folder = self.root/'outputs/audio'; folder.mkdir(parents=True,exist_ok=True)
        request_file = folder/f'.{job_id}-request.json'
        result_file = folder/f'.{job_id}-result.json'
        output = folder/f'{job_id}.wav'
        request_file.write_text(json.dumps(payload)); request_file.chmod(0o600)
        try:
            result = await self.manager.run_owned('ace-music', [str(python),str(worker),
                '--project',str(project),'--request',str(request_file),'--output',str(output),
                '--result',str(result_file)], cwd=project)
            if result.returncode or not result_file.is_file() or not output.is_file():
                log = (result.stderr or b'')[-3000:].decode(errors='replace')
                raise RuntimeError('ACE-Step generation failed: '+log[-1200:])
            measured = json.loads(result_file.read_text())
            original = None
            if payload.get('source_audio'):
                source = Path(payload['source_audio'])
                original = folder/f'{job_id}-original{source.suffix}'
                shutil.copy2(source,original)
            return {'artifact':str(output),'content_type':'audio/wav',
                    'original_artifact_path':str(original) if original else None,
                    'original_content_type':measured.get('original_content_type'),
                    'actual_duration_seconds':measured['duration'],
                    'generation_seconds':measured['generation_seconds'],
                    'peak_rss_gib':measured['peak_rss_gib'], 'peak_metal_gib':measured.get('peak_metal_gib'),
                    'music_quality_json':measured.get('music_quality_json'), 'music_model':payload['model']}
        finally:
            request_file.unlink(missing_ok=True); result_file.unlink(missing_ok=True)
