"""Hash the frozen TabSyn sample grid; reads no benchmark partition or model."""
import argparse
import hashlib
import json
import os
from pathlib import Path

FITS_SHA256 = 'cb2c1c134ccd1af9865e156faee8796143c212d1f76e70ab70f7048f82a08272'
CONFIG_SHA256 = 'a34729383eac6783c7e754a791094054d54f1062655db9a306cd4cca296c5ebc'
SAMPLES = Path('/home/ubuntu/dope-scratch-x1/dope-rf-tabsyn-sample-v1')


def capture(fit_bytes, samples=SAMPLES):
    if hashlib.sha256(fit_bytes).hexdigest() != FITS_SHA256:
        raise ValueError('frozen fit registry digest differs')
    fits = json.loads(fit_bytes)
    if fits['official_tests_opened'] is not False or fits['config_sha256'] != CONFIG_SHA256:
        raise ValueError('frozen fit registry test flag or configuration refused')
    records = []
    for fit in sorted(fits['cells'], key=lambda c: c['dataset_id']):
        if fit['status'] != 'fit_ok' or fit['actual_fit_exit'] != 0 or fit['fit_seed'] != 11:
            raise ValueError('frozen fit identity differs')
        dataset = fit['dataset_id']
        if len(dataset) != 16 or any(c not in '0123456789abcdef' for c in dataset):
            raise ValueError('sample identity is outside the frozen registry')
        for size in (1, 4):
            for seed in (101, 211, 307):
                path = samples / dataset / f'sample-{seed}-{size}n.csv'
                if path.is_symlink() or path.resolve() != samples.resolve() / dataset / path.name:
                    raise ValueError('redirected sample refused')
                row = {'dataset': dataset, 'size': size, 'sample_seed': seed,
                       'official_tests_opened': False, 'formal_dp': False}
                if path.is_file():
                    raw = path.read_bytes()
                    row.update(status='ok', reason=None, bytes=len(raw),
                               synthetic_sha256=hashlib.sha256(raw).hexdigest())
                else:
                    row.update(status='unavailable', reason='sample_csv_absent',
                               bytes=None, synthetic_sha256=None)
                records.append(row)
    return {'format': 'dope-wave2-tabsyn-sample-evidence-v1',
            'official_tests_opened': False, 'formal_dp': False,
            'fit_registry_sha256': FITS_SHA256, 'cells': records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fits', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.uname().nodename != 'xbabe1':
        raise ValueError('sample evidence capture is restricted to xbabe1')
    if args.fits.is_symlink():
        raise ValueError('redirected fit registry refused')
    result = capture(args.fits.read_bytes())
    result['producer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('sample receipts', len(result['cells']))


if __name__ == '__main__':
    main()
