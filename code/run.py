"""Dispatch source modules without empty __init__.py files."""
from pathlib import Path
import runpy
import sys
import types

def main():
    if len(sys.argv)<2:
        raise SystemExit('Usage: python code/run.py result.rebuild --output-dir <directory>')
    name=sys.argv[1]
    if name.startswith('mcm_code.'): name=name[len('mcm_code.'):]
    if not name.startswith(('result.','table.','figure.','check.','common.')):
        raise SystemExit('Unknown source-module group')
    package=types.ModuleType('mcm_code')
    package.__path__=[str(Path(__file__).resolve().parent)]
    package.__package__='mcm_code'
    sys.modules['mcm_code']=package
    sys.argv=[sys.argv[0],*sys.argv[2:]]
    runpy.run_module('mcm_code.'+name,run_name='__main__')

if __name__=='__main__': main()
