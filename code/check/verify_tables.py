"""Compare regenerated paper tables with the preserved submission snapshots."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..common.paths import SUPPORT_ROOT, protected_output_path


LABEL_ALIASES = {
    'model': {'A': '版本 A', '完全信息基准（A）': '版本 A'},
    'strategy': {'bfreeze：冻结储能滚动调整': 'bfreeze：固定储能滚动调整'},
}


def compare(actual, reference):
    a = pd.read_csv(actual)
    b = pd.read_csv(reference)
    # Runtime is nondeterministic; stable semantic labels remain comparison keys.
    common = [c for c in b.columns if c != 'solver_seconds']
    missing = sorted(set(common)-set(a.columns))
    extra = sorted(set(a.columns)-set(common)-{'solver_seconds'})
    if missing or extra:
        return dict(status='FAIL', missing_columns=missing, extra_columns=extra)
    a, b = a[common].copy(), b[common].copy()
    keys = []
    for name in common:
        if a[name].dtype == object or b[name].dtype == object:
            a[name] = a[name].fillna('').astype(str).str.replace('--','-',regex=False)
            b[name] = b[name].fillna('').astype(str).str.replace('--','-',regex=False)
            if name in LABEL_ALIASES:
                a[name] = a[name].replace(LABEL_ALIASES[name])
                b[name] = b[name].replace(LABEL_ALIASES[name])
            keys.append(name)
    if a.shape != b.shape:
        return dict(status='FAIL', actual_shape=list(a.shape), reference_shape=list(b.shape))
    if keys:
        if a.duplicated(keys).any() or b.duplicated(keys).any():
            return dict(status='FAIL', reason='Duplicate semantic row keys', keys=keys)
        a = a.sort_values(keys).reset_index(drop=True)
        b = b.sort_values(keys).reset_index(drop=True)
        if not a[keys].equals(b[keys]):
            return dict(status='DIFFERENT', reason='Semantic row keys differ', keys=keys)
    differences = {}
    for name in common:
        if pd.api.types.is_numeric_dtype(a[name]) and pd.api.types.is_numeric_dtype(b[name]):
            x,y = a[name].to_numpy(dtype=float),b[name].to_numpy(dtype=float)
            mask = np.isclose(x,y,atol=1e-5,rtol=1e-8,equal_nan=True)
            if not mask.all():
                differences[name] = dict(rows=np.flatnonzero(~mask).tolist()[:12], max_absolute_difference=float(np.nanmax(np.abs(x-y))))
        elif not a[name].equals(b[name]):
            differences[name] = {'rows': a.index[a[name] != b[name]].tolist()[:12]}
    return dict(status='PASS' if not differences else 'DIFFERENT', rows=len(a), differences=differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=SUPPORT_ROOT/'table')
    parser.add_argument('--report',type=Path,required=True)
    args = parser.parse_args()
    report_path = protected_output_path(args.report)
    report = {'tables':{},'scope':'Numeric and semantic-key comparison of 19 paper tables; only solver runtime and row order excluded; documented label aliases normalized.'}
    for reference in sorted(args.reference.glob('*.csv')):
        actual = args.root/reference.name
        report['tables'][reference.name] = compare(actual,reference) if actual.is_file() else dict(status='FAIL',reason='Missing regenerated table')
    report['status'] = 'PASS' if len(report['tables']) == 19 and all(x['status']=='PASS' for x in report['tables'].values()) else 'FAIL'
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
