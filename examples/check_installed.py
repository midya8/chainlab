"""Check the wheel outside the checkout, including every test and CLI report."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV.pop('PYTHONPATH', None)
MODULES = ['chainlab', 'chainlab.__main__', 'chainlab.cli', 'chainlab.core',
           'chainlab.demo', 'chainlab.evidence', 'chainlab.evidence_export',
           'chainlab.occupancy', 'chainlab.online', 'chainlab.path_verify',
           'chainlab.pathrisk']
COMMANDS = [[name] for name in ['demo', 'online-demo', 'evidence-report']]
COMMANDS += [[name, '--input', 'request.json'] for name in ['infer', 'viterbi', 'filter']]
COMMANDS += [['forecast', '--input', 'request.json', '--horizon', '3'],
             ['fit', '--input', 'fit.json', '--iterations', '2'],
             ['evidence', '--input', 'request.json', '--cut', '6']]


def call(args, cwd, env=ENV):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True)


with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    wheels = work / 'wheels'
    wheels.mkdir()
    call([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '-w', str(wheels),
          str(ROOT)], work)
    built = list(wheels.glob('*.whl'))
    assert len(built) == 1, built
    venv.create(work / 'env', with_pip=True)
    py = str(work / 'env/bin/python')
    call([py, '-m', 'pip', 'install', '--no-deps', str(built[0])], work)
    outside = work / 'outside'
    outside.mkdir()
    for name in ['tests', 'examples', 'assets']:
        shutil.copytree(ROOT / name, outside / name)
    # Some tests locate JSON fixtures relative to their copied test directory.
    shutil.copytree(ROOT / 'chainlab/artifacts', outside / 'chainlab/artifacts')
    # One static test reads the installed verifier's source as text. There is no
    # __init__.py here, so this directory cannot replace the installed package.
    verifier = call([py, '-I', '-c',
                     'import chainlab.path_verify;print(chainlab.path_verify.__file__)'],
                    outside).strip()
    shutil.copyfile(verifier, outside / 'chainlab/path_verify.py')
    suite = '''
import importlib,importlib.util,json,pathlib,sys,unittest
outside=pathlib.Path.cwd().resolve()
sys.path.insert(0,str(outside))
names=json.loads(sys.argv[1])
source=pathlib.Path(sys.argv[2]).resolve()
prefix=pathlib.Path(sys.prefix).resolve()
for name in names:importlib.import_module(name)
def check():
    for name in names:
        origin=pathlib.Path(importlib.util.find_spec(name).origin).resolve()
        assert origin.is_relative_to(prefix),(name,str(origin))
        assert not origin.is_relative_to(source),(name,str(origin))
        assert 'site-packages' in origin.parts,(name,str(origin))
check()
tests=unittest.defaultTestLoader.discover(str(outside/'tests'))
result=unittest.TextTestRunner(verbosity=2).run(tests)
assert result.testsRun>0 and result.wasSuccessful()
check()
print(json.dumps(dict(tests=result.testsRun,installed_origins=len(names),
                     checked_before_and_after=True)))
'''
    print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT)], outside))
    reference = work / 'reference'
    reference.mkdir()
    source_env = ENV.copy()
    source_env['PYTHONPATH'] = str(ROOT)
    saved = json.loads(call([sys.executable, '-m', 'chainlab', 'evidence-report'],
                            reference, source_env))
    trained = saved['arms']['trained']
    for folder in [reference, outside]:
        (folder / 'request.json').write_text(json.dumps(dict(
            model=trained['model'], sequence=trained['sequence'])))
        (folder / 'fit.json').write_text(json.dumps(dict(
            model=trained['model'], sequences=[trained['sequence']])))
    for args in COMMANDS:
        # Compare complete parsed reports, not just lengths or selected metrics.
        expected = json.loads(call([sys.executable, '-m', 'chainlab'] + args,
                                   reference, source_env))
        for entry in [[py, '-I', '-m', 'chainlab'], [str(work / 'env/bin/chainlab')]]:
            assert json.loads(call(entry + args, outside)) == expected, (entry, args)
    # The renderer writes its result, so preserve and compare the complete bytes.
    figure = outside / 'assets/sequence-audit.svg'
    previous = figure.read_bytes()
    installed_env = ENV.copy()
    installed_env['PATH'] = str(work / 'env/bin') + os.pathsep + installed_env['PATH']
    installed_env['PYTHONPATH'] = str(outside)
    call([py, str(outside / 'examples/render_demo.py')], outside, installed_env)
    assert figure.read_bytes() == previous
    # Keep the existing exact path, occupancy and forgery installation checks.
    import sysconfig
    installed_path = call([py, '-I', '-c',
                           'import sysconfig;print(sysconfig.get_path("purelib"))'],
                          outside).strip()
    print(call(['bash', str(outside / 'examples/verify_wheel.sh'), installed_path],
               outside, installed_env))
    print(json.dumps(dict(complete_reports=len(COMMANDS), entrypoints=2,
                          reproduced_figures=1)))
