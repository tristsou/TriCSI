"""Create and verify a complete, unmodified research snapshot using stdlib only."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import platform
import stat
import subprocess
import sys
import zipfile

EXCLUDED_DIRS = {'.git', '.venv', 'venv', '__pycache__', 'node_modules', '.pytest_cache', '.mypy_cache'}
EXCLUDED_SUFFIXES = {'.zip', '.pyc', '.pyo', '.pem', '.key'}
REQUIRED = ['train_tricsi_anchor_final.py', 'train_csi_cnn.py', 'csi_io.py', 'project_config.py',
            'models/tricsi_anchor_s4_s5_s6_s7.pt',
            *[f'data/raw/{s}' for s in ['S4','S5','S6','S7','S8','S9']]]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def environment():
    versions = {}
    for name in ['numpy','pandas','scipy','scikit-learn','matplotlib','joblib','torch','pyserial']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  machine=platform.machine(), packages=versions,
                  note='Environment running the bundle tool; it is not proof of the environment used for earlier training.')
    try:
        freeze = subprocess.run([sys.executable,'-m','pip','freeze'], capture_output=True,
                                text=True, timeout=45, check=True).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        freeze = None
        result['pip_freeze_error'] = type(exc).__name__
    return result, freeze


def eligible(relative):
    return (not any(part in EXCLUDED_DIRS for part in relative.parts)
            and relative.suffix.lower() not in EXCLUDED_SUFFIXES
            and not any(part == '.env' or part.startswith('.env.') for part in relative.parts))


def role(name):
    if name.startswith(('data/raw/','data_3tx/raw/')):
        return 'raw_recording'
    if name.startswith(('models/','checkpoints/')) or name.endswith(('.pt','.pth','.joblib')):
        return 'fitted_model'
    if name.startswith('outputs/'):
        return 'prepared_input'
    if name.startswith('results/'):
        return 'experiment_output'
    if name.endswith(('.py','.ps1','.bat','.c','.cpp','.h','.ino')):
        return 'source_code'
    return 'documentation_or_metadata'


def create_bundle(project, output):
    root, output = Path(project).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise ValueError(f'Project directory missing: {root}')
    if output == root or root in output.parents:
        raise ValueError('Write the ZIP outside the project to avoid recursive packaging.')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    missing = [s for s in REQUIRED if not (root/s).exists()]
    empty_sessions = [s for s in REQUIRED if (root/s).is_dir() and not list((root/s).glob('*.csv'))]
    if missing or empty_sessions:
        raise ValueError(f'Missing required two-link assets: {missing}; empty sessions: {empty_sessions}. '
                         'Use the original working project, not a code-only GitHub checkout.')
    paths, omitted = [], []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if not eligible(relative):
            continue
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root):
            omitted.append(dict(path=relative.as_posix(), reason='symlink_not_followed'))
            continue
        if path.is_file():
            paths.append(path)
    env, freeze = environment()
    if freeze is None:
        raise ValueError('Could not capture pip freeze; run with the working project Python environment.')
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    created = False
    try:
        with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            created = True
            for i, path in enumerate(paths, 1):
                relative = path.relative_to(root).as_posix()
                name = 'project/' + relative
                before = path.stat()
                h = hashlib.sha256()
                with path.open('rb') as src, archive.open(name, 'w', force_zip64=True) as dst:
                    for chunk in iter(lambda: src.read(1024*1024), b''):
                        dst.write(chunk); h.update(chunk)
                after = path.stat()
                if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                    raise ValueError(f'File changed during packaging: {relative}. Finish recording/training first.')
                rows.append(dict(path=name, bytes=after.st_size, sha256=h.hexdigest(), role=role(relative)))
                if i % 100 == 0 or i == len(paths):
                    print(f'Packaged {i}/{len(paths)} files', flush=True)
            for name, payload in [('environment.json', json.dumps(env,indent=2).encode()),
                                  ('requirements-lock.txt', freeze.encode())]:
                archive.writestr(name,payload)
                rows.append(dict(path=name,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),role='environment'))
            manifest = dict(format_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                project_name=root.name, files=rows, omitted=omitted,
                excluded='Virtual environments, Git history, old ZIPs, bytecode, .env and key files',
                required_two_link_assets_present=True,
                three_link_sessions=[p.name for p in sorted((root/'data_3tx/raw').glob('3TX_*')) if p.is_dir()],
                limitations='Byte-preserving snapshot, not validation of labels, historical model identity, or experimental claims. Original MAC values in recordings and source are retained.')
            archive.writestr('manifest.json',json.dumps(manifest,indent=2).encode())
        verify_bundle(output)
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        raise
    print(f'Reproducibility bundle: {output}')
    print(f'SHA256: {digest(output)}')
    return manifest


def verify_bundle(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names)!=len(set(names)):
            raise ValueError('Duplicate archive members.')
        for info in archive.infolist():
            name = info.filename
            parts = PurePosixPath(name).parts
            if (PurePosixPath(name).is_absolute() or '..' in parts or '\\' in name
                    or any(':' in p for p in parts) or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError(f'Unsafe archive path: {name}')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest.get('format_version') != 1:
            raise ValueError('Unsupported bundle format.')
        rows = manifest['files']
        expected = [row['path'] for row in rows]
        if len(expected)!=len(set(expected)) or set(names)!=set(expected)|{'manifest.json'}:
            raise ValueError('Archive members do not match the manifest.')
        for row in rows:
            h, size = hashlib.sha256(), 0
            with archive.open(row['path']) as src:
                for chunk in iter(lambda: src.read(1024*1024), b''):
                    h.update(chunk); size+=len(chunk)
            if h.hexdigest()!=row['sha256'] or size!=row['bytes']:
                raise ValueError(f"Integrity failure: {row['path']}")
    print(f'VERIFIED: {len(rows)} file hashes match the bundle manifest.')
    return manifest
