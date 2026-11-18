"""One reader-facing command for the supported TriCSI workflows."""
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
ROUTES = {
    'inventory': ('module', 'experiments.session_robustness', ['inventory']),
    'prepare': ('module', 'experiments.session_robustness', ['prepare']),
    'develop': ('module', 'experiments.session_robustness', ['develop']),
    'freeze': ('module', 'experiments.session_robustness', ['freeze']),
    'evaluate': ('module', 'experiments.session_robustness', ['evaluate']),
    'diagnose': ('module', 'experiments.session_diagnostics', ['audit']),
    'compare-shape': ('module', 'experiments.session_diagnostics', ['compare']),
    'reconstruct': ('module', 'experiments.session_diagnostics.historical', []),
    'two-evaluate': ('script', 'evaluate_frozen_anchor_confirmatory.py', []),
    'three-train': ('script', 'tricsi_3tx_modeling_bundle/compare_and_freeze_3tx.py', []),
    'three-evaluate': ('script', 'tricsi_3tx_modeling_bundle/evaluate_frozen_3tx.py', []),
}


def dispatch(command, arguments, *, project_root=ROOT, link_map=None, results_root=None):
    """Run with the original working-directory convention and source bytes."""
    root = Path(project_root).resolve()
    kind, target, prefix = ROUTES[command]
    source = root / target if kind == 'script' else root / (target.replace('.', '/') + '.py')
    if kind == 'module' and not source.is_file():
        source = root / target.replace('.', '/') / '__main__.py'
    if not source.is_file():
        raise ValueError(f'Workflow source missing: {source}')
    sys.path[:0] = [str(root), str(source.parent), str(root/'legacy/scripts')]
    if results_root is not None and command != 'two-evaluate':
        raise ValueError('--results-root applies only to two-evaluate; other workflows use --output.')
    if command == 'two-evaluate' and not any(x in arguments for x in ['--help','-h']):
        config = importlib.import_module('project_config')
        destination = Path(results_root).resolve() if results_root else root/'results/reproduced_two_link'
        if (destination/'tricsi_anchor_confirmatory').exists():
            raise FileExistsError('Evaluation output already exists. Choose a new --results-root.')
        config.RESULTS_DIR = destination
    if link_map is not None:
        if not command.startswith('three-'):
            raise ValueError('--link-map is supported only for three-link workflows.')
        mapping = json.loads(Path(link_map).read_text(encoding='utf-8-sig'))
        module = importlib.import_module('tricsi_3tx_modeling')
        expected = set(module.LINK_TO_MAC)
        if set(mapping) != expected or any(not isinstance(v, str) for v in mapping.values()):
            raise ValueError(f'Link map must contain exactly {sorted(expected)}.')
        import re
        mapping = {k: module.normalize_mac(v) for k, v in mapping.items()}
        if len(set(mapping.values())) != 3 or any(not re.fullmatch(r'(?:[0-9A-F]{2}:){5}[0-9A-F]{2}', v) for v in mapping.values()):
            raise ValueError('Link map must contain three distinct valid MAC addresses.')
        module.LINK_TO_MAC = mapping
    old_args, old_cwd = sys.argv, Path.cwd()
    try:
        os.chdir(root)
        sys.argv = [str(source), *prefix, *arguments]
        if kind == 'module':
            runpy.run_module(target, run_name='__main__')
        else:
            runpy.run_path(str(source), run_name='__main__')
    finally:
        sys.argv = old_args
        os.chdir(old_cwd)


def doctor(root, code_only=False):
    from .bundle import digest
    root = Path(root).resolve()
    status = []
    for module, package in [('numpy','numpy'), ('pandas','pandas'), ('scipy','scipy'),
                            ('sklearn','scikit-learn'), ('matplotlib','matplotlib'),
                            ('joblib','joblib'), ('torch','torch')]:
        found = importlib.util.find_spec(module) is not None
        status.append((found, f'Python package: {package}'))
    records = json.loads((root/'metadata/engine_hashes.json').read_text())
    for name, expected in records.items():
        path = root / name
        status.append((path.is_file() and digest(path)==expected, f'Preserved engine: {name}'))
    if not code_only:
        for name in ['data/raw/S4','data/raw/S5','data/raw/S6','data/raw/S7',
                     'data/raw/S8','data/raw/S9','models/tricsi_anchor_s4_s5_s6_s7.pt']:
            status.append(((root/name).exists(), f'Local asset: {name}'))
    for ok, message in status:
        print(f"{'OK' if ok else 'MISSING/CHANGED'}  {message}")
    if not code_only:
        print('Existence is not historical identity. Verify the bundle manifest and recorded run hashes.')
    return all(ok for ok, _ in status)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument('command', choices=[*ROUTES, 'run', 'doctor', 'bundle', 'verify-bundle'])
    p.add_argument('--project-root', type=Path, default=ROOT,
                   help='Project whose data and exact code should be used (default: this checkout)')
    p.add_argument('--link-map', type=Path, help='Local JSON mapping for the three physical transmitter MACs')
    p.add_argument('--results-root', type=Path, help='New output root for the historical two-link evaluator')
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv in (['--help'], ['-h']):
        p.print_help()
        return
    args, rest = p.parse_known_args(argv)
    if rest[:1] == ['--']:
        rest = rest[1:]
    try:
        if args.command == 'run':
            q = argparse.ArgumentParser()
            q.add_argument('recipe',type=Path)
            q.add_argument('--dry-run',action='store_true')
            opts = q.parse_args(rest)
            recipe = json.loads(opts.recipe.read_text())
            if recipe.get('command') not in ROUTES or not isinstance(recipe.get('arguments'),list) or not all(isinstance(x,str) for x in recipe['arguments']):
                raise ValueError('Recipe must contain a supported command and a list of string arguments.')
            print(json.dumps(recipe,indent=2))
            if not opts.dry_run:
                dispatch(recipe['command'], recipe['arguments'],project_root=args.project_root,
                         link_map=args.link_map,results_root=args.results_root)
        elif args.command == 'doctor':
            q = argparse.ArgumentParser(); q.add_argument('--code-only', action='store_true')
            opts = q.parse_args(rest)
            if not doctor(args.project_root, opts.code_only):
                raise SystemExit(1)
        elif args.command == 'bundle':
            from .bundle import create_bundle
            q = argparse.ArgumentParser(); q.add_argument('--output', type=Path, required=True)
            opts = q.parse_args(rest)
            create_bundle(args.project_root, opts.output)
        elif args.command == 'verify-bundle':
            from .bundle import verify_bundle
            q = argparse.ArgumentParser(); q.add_argument('archive', type=Path)
            verify_bundle(q.parse_args(rest).archive)
        else:
            dispatch(args.command, rest, project_root=args.project_root, link_map=args.link_map,
                     results_root=args.results_root)
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f'STOP: {exc}') from exc


if __name__ == '__main__':
    main()
