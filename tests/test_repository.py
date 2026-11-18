"""Software contracts only. Synthetic fixtures are never research evidence."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tricsi.__main__ import ROOT, ROUTES, dispatch
from tricsi.bundle import REQUIRED, create_bundle, digest, verify_bundle


class RepositoryTests(unittest.TestCase):
    def test_all_moves_and_protected_engine_bytes_match(self):
        for row in json.loads((ROOT/'metadata/migration.json').read_text())['files']:
            self.assertTrue((ROOT/row['new']).is_file(), row['new'])
            self.assertEqual(digest(ROOT/row['new']),row['sha256'],row['new'])
            self.assertFalse((ROOT/row['old']).exists(),row['old'])
        for name, expected in json.loads((ROOT/'metadata/engine_hashes.json').read_text()).items():
            self.assertEqual(digest(ROOT/name),expected,name)

    def test_all_supported_workflows_have_help(self):
        for command in ROUTES:
            result = subprocess.run([sys.executable,'-m','tricsi',command,'--help'],
                                    cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,(command,result.stderr))
            self.assertIn('usage:',result.stdout)
            # This must be the delegated parser, not only the top-level CLI.
            self.assertNotIn('One reader-facing command',result.stdout)

    def test_recipes_dry_run_without_creating_results(self):
        for recipe in (ROOT/'configs').glob('*.json'):
            data = json.loads(recipe.read_text())
            self.assertIn(data['command'],ROUTES)
            result = subprocess.run([sys.executable,'-m','tricsi','run',str(recipe),'--dry-run'],
                                    cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout),data)

    def test_maintained_document_links_resolve(self):
        paths = [ROOT/'README.md',ROOT/'CONTRIBUTING.md',ROOT/'legacy/README.md',ROOT/'archive/README.md',
                 *[ROOT/'docs'/s for s in ['README.md','SETUP.md','DATA.md','OVERVIEW.md',
                     'REPRODUCE.md','ARCHITECTURE.md','PUBLISHING.md','AVAILABILITY.md']],
                 ROOT/'metadata/README.md',ROOT/'configs/README.md',ROOT/'reports/README.md']
        for path in paths:
            for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
                if target.startswith(('http:','https:','#')):
                    continue
                self.assertTrue((path.parent/target.split('#')[0]).exists(),(path,target))

    def test_bundle_roundtrip_exclusions_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            temp=Path(directory);project=temp/'working';project.mkdir()
            for name in REQUIRED:
                path=project/name
                if name.startswith('data/raw/'):
                    path.mkdir(parents=True);(path/'trial.csv').write_text('synthetic fixture\n')
                else:
                    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
            (project/'.env').write_text('secret fixture')
            (project/'old.zip').write_bytes(b'old archive')
            (project/'.venv').mkdir();(project/'.venv/skip').write_text('skip')
            out=temp/'reproduction.zip'
            with patch('tricsi.bundle.environment',return_value=({'fixture':True},'numpy==0\n')):
                manifest=create_bundle(project,out)
            self.assertTrue(manifest['required_two_link_assets_present'])
            with zipfile.ZipFile(out) as archive:
                self.assertNotIn('project/.env',archive.namelist())
                self.assertNotIn('project/old.zip',archive.namelist())
                self.assertNotIn('project/.venv/skip',archive.namelist())
                original={i.filename:archive.read(i.filename) for i in archive.infolist()}
            self.assertEqual(verify_bundle(out)['files'],manifest['files'])
            changed=temp/'changed.zip'
            original['project/project_config.py']=b'tampered'
            with zipfile.ZipFile(changed,'w') as archive:
                for name, content in original.items():archive.writestr(name,content)
            with self.assertRaises(ValueError):verify_bundle(changed)
            with self.assertRaises(FileExistsError):create_bundle(project,out)
            with self.assertRaises(ValueError):create_bundle(project,project/'recursive.zip')

    def test_missing_assets_and_unsafe_archive_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(ValueError):create_bundle(root,root.parent/'uncreated.zip')
            path=root/'unsafe.zip'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('../outside',b'bad')
                archive.writestr('manifest.json','{}')
            with self.assertRaises(ValueError):verify_bundle(path)

    def test_local_link_mapping_does_not_rewrite_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            mapping={'back_left':'AA:00:00:00:00:01','middle_back':'AA:00:00:00:00:04','back_right':'AA:00:00:00:00:03'}
            path=Path(directory)/'mapping.json';path.write_text(json.dumps(mapping))
            engine=ROOT/'tricsi_3tx_modeling_bundle/tricsi_3tx_modeling.py'
            before=digest(engine)
            def inspect(*args,**kwargs):
                self.assertEqual(sys.modules['tricsi_3tx_modeling'].LINK_TO_MAC,mapping)
            try:
                with patch('tricsi.__main__.runpy.run_path',side_effect=inspect):
                    dispatch('three-evaluate',['--help'],link_map=path)
                self.assertEqual(digest(engine),before)
                path.write_text('{}')
                with self.assertRaises(ValueError):dispatch('three-evaluate',['--help'],link_map=path)
            finally:
                sys.modules.pop('tricsi_3tx_modeling',None)


if __name__=='__main__':
    unittest.main()
