"""Read-only live snapshot of this reproduction, including actual process handles."""
import json
from pathlib import Path
import subprocess

BASE = Path('/root/genrec-reproduction')

def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def log_tail(path):
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 12000))
        lines = stream.read().decode('utf-8', 'replace').replace('\r', '\n').splitlines()
    lines = [line for line in lines if line.strip()]
    return {'bytes': path.stat().st_size, 'mtime': path.stat().st_mtime,
            'last_lines': lines[-10:]}

status = read_json(BASE / 'status.json')
processes = []
for process in Path('/proc').iterdir():
    if not process.name.isdigit():
        continue
    try:
        arguments = process.joinpath('cmdline').read_bytes().split(b'\0')
        command = [argument.decode('utf-8', 'replace') for argument in arguments if argument]
        if any(argument in {str(BASE / filename) for filename in
                            ['pipeline.py', 'run_stage.py', 'evaluate_tiger.py',
                             'finish_environment.sh']} for argument in command):
            parent_line = next(line for line in process.joinpath('status').read_text().splitlines()
                               if line.startswith('PPid:'))
            processes.append({'pid': int(process.name), 'ppid': int(parent_line.split()[1]),
                              'argv': command})
    except (OSError, ProcessLookupError):
        pass

by_pid = {process['pid']: process for process in processes}
workers = [process for process in processes if process['ppid'] in by_pid and
           by_pid[process['ppid']]['argv'] == process['argv']]
roots = [process for process in processes if process not in workers]
snapshot = {'status': status, 'processes': roots, 'dataloader_worker_count': len(workers),
            'manifests': {name: read_json(BASE / 'manifests' / name) for name in
                          ['environment-status.json', 'data-status.json', 'model-status.json']},
            'results': [path.name for path in (BASE / 'results').glob('*.json')],
            'pipeline_log': log_tail(BASE / 'logs/pipeline.log')}
for name in ['data-status.json', 'model-status.json']:
    record = snapshot['manifests'][name]
    if record and 'files' in record:
        snapshot['manifests'][name] = {'state': record['state'], 'file_count': len(record['files'])}
if status and status.get('split') and status.get('stage'):
    snapshot['stage_log'] = log_tail(BASE / 'logs' / (status['split'] + '-' + status['stage'] + '.log'))
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu',
                      '--format=csv,noheader'], capture_output=True, text=True)
snapshot['gpu'] = gpu.stdout.strip() or gpu.stderr.strip()
print(json.dumps(snapshot, indent=2), flush=True)
