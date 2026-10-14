#!/usr/bin/env python3
"""One authenticated in-process paper build, check, or direct PDF admission."""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / 'docs/whitepaper'
SPEC = importlib.util.spec_from_file_location('paper_review_outputs', REPO / 'ops/ci/paper-review-outputs.py')
REVIEW = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REVIEW
SPEC.loader.exec_module(REVIEW)
PDF_STEMS = ('dope-mfs', 'supplement', 'dope-mfs-anonymous', 'supplement-anonymous')
OUTPUT_PATHS = ('docs/whitepaper/generated', 'docs/whitepaper/figures', 'docs/readme', 'README.md',
                *(f'docs/whitepaper/{stem}.pdf' for stem in PDF_STEMS))
COMPILER_OUTPUTS = ('docs/whitepaper/generated/architecture-figure-hashes.json',
                    'docs/whitepaper/figures/architecture.pdf',
                    *(f'docs/whitepaper/{stem}.pdf' for stem in PDF_STEMS))


def capacity_profile():
    """Record affinity and cgroup limits; logical CPU count is not a quota."""
    profile = {'hostname': os.uname().nodename, 'logical_cpus': os.cpu_count(),
               'affinity_cpus': sorted(os.sched_getaffinity(0)), 'cgroup_limits': {},
               'mem_total_kib': next((int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                                      if line.startswith('MemTotal:')), None)}
    # Include ancestors: a child may report 'max' under a bounded parent.
    cgroup = next((line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
                   if line.startswith('0::')), '/')
    root = Path('/sys/fs/cgroup')
    directory = root / cgroup.lstrip('/')
    while directory.is_relative_to(root):
        for name in ('cpu.max', 'cpuset.cpus.effective', 'memory.max', 'memory.current'):
            path = directory / name
            if path.is_file():
                profile['cgroup_limits'][str(path)] = path.read_text().strip()
        if directory == root:
            break
        directory = directory.parent
    return profile


class BuildReceipt:
    def __init__(self, mode):
        self.directory = REPO / 'target/paper-clean' / ('attempt-' + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.payload = {'format': 'dope-paper-budget-repair-v1', 'mode': mode,
                        'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        'attempt': self.directory.name, 'profile': capacity_profile(), 'stages': [],
                        'status': 'running', 'cold_budget': 'unmeasured',
                        'historical_failure': {'run_id': 38489720791, 'job_id': 115612120522,
                                               'outcome': 'cancelled_after_35_minute_limit'},
                        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                        'tree': subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=REPO, text=True).strip(),
                        'working_diff_sha256': hashlib.sha256(subprocess.check_output(
                            ['git', 'diff', 'HEAD', '--'], cwd=REPO)).hexdigest()}
        self.save()

    def save(self):
        raw = json.dumps(self.payload, indent=2, sort_keys=True) + '\n'
        (self.directory / 'receipt.json').write_text(raw)
        (self.directory.parent / 'budget-repair.json').write_text(raw)

    def stage(self, name, action):
        start = time.monotonic()
        before = [resource.getrusage(who) for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
        entry = {'name': name, 'exit': None}
        self.payload['stages'].append(entry)
        self.save()
        try:
            result = action()
            entry['exit'] = 0
            return result
        except BaseException as error:
            entry['exit'] = getattr(error, 'returncode', 1)
            entry['error_type'] = type(error).__name__
            self.payload['status'] = 'failed'
            raise
        finally:
            after = [resource.getrusage(who) for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
            entry.update(elapsed_seconds=time.monotonic() - start,
                         cpu_seconds=sum(a.ru_utime + a.ru_stime - b.ru_utime - b.ru_stime
                                         for a, b in zip(after, before)),
                         self_peak_rss_kib=after[0].ru_maxrss,
                         max_completed_child_peak_rss_kib=after[1].ru_maxrss)
            render_commands = self.directory / 'render-commands.json'
            if render_commands.is_file():
                self.payload['render_commands'] = json.loads(render_commands.read_text())
            self.save()

    def bind(self, verified):
        for key in ('inputs', 'code', 'consumed', 'expected', 'auxiliary_expected'):
            self.payload[key + '_sha256'] = {name: hashlib.sha256(raw).hexdigest()
                                            for name, raw in getattr(verified, key)}
        self.save()


def run(command, receipt, *, cwd=None, env=None):
    if cwd is None:
        cwd = REPO
    index = len(receipt.payload.setdefault('commands', []))
    log = receipt.directory / f'command-{index:03d}.log'
    entry = {'command': list(map(str, command)), 'cwd': str(cwd), 'log': str(log), 'exit': None}
    receipt.payload['commands'].append(entry)
    receipt.save()
    print(' '.join(entry['command']), flush=True)
    with log.open('wb') as output:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
    entry['exit'] = result.returncode
    receipt.save()
    result.check_returncode()


def output_drift(receipt, *, before_compile=False):
    paths = OUTPUT_PATHS
    if before_compile:
        # A cold clean removes the architecture outputs. They are derived by
        # the internal compiler and enter the full zero-drift check afterward.
        paths += tuple(':(exclude)' + name for name in COMPILER_OUTPUTS)
    run(['git', 'diff', '--exit-code', '--', *paths], receipt)
    # git diff does not see unexpected untracked outputs.
    extra = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', *paths],
                                    cwd=REPO, text=True)
    if extra.strip():
        raise ValueError('untracked paper output drift')


def auxiliary_outputs(verified, *, check, receipt):
    verified.revalidate()
    REVIEW.render_review_outputs(verified, check=True)
    if {name for name, _ in verified.auxiliary_expected} != set(REVIEW.AUXILIARY_OUTPUTS):
        raise ValueError('incomplete expected figure/README inventory')
    for name, _ in verified.auxiliary_expected:
        if (REPO / name).resolve() != REPO.resolve() / name:
            raise ValueError('redirected figure/README output')
    for name, raw in verified.auxiliary_expected:
        if check:
            if REVIEW.safe_bytes(REPO, name) != raw:
                raise ValueError('figure/README output differs from expected rendering')
        else:
            path = REPO / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    verified.revalidate()
    return verified


def anonymous_sources(root, *, paper):
    spec = importlib.util.spec_from_file_location('paper_anonymity', paper / 'scripts/public_hardware.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    strip_journal_availability = helper.strip_journal_availability
    old = ('\\author{Jepson Taylor$^1$, Alton Alexander$^1$\\\\[0.15cm]\n'
           '$^1$NEVERHUMAN Research}')
    new = '\\author{Anonymous\\\\[0.15cm]\nAffiliation withheld for review.}'
    paths = {}
    for stem in ('dope-mfs', 'supplement'):
        text = (paper / f'{stem}.tex').read_text()
        if text.count(old) != 1:
            raise ValueError('paper author marker is not unique')
        text = text.replace(old, new, 1)
        if stem == 'dope-mfs':
            text = strip_journal_availability(text)
        if 'neverhuman' in text.lower():
            raise ValueError('anonymous paper still names the repository')
        path = root / f'{stem}-anonymous.tex'
        path.write_text(text)
        paths[stem + '-anonymous'] = path
    return paths


def compile_pdfs(verified, receipt):
    """Internal compiler. Public entry points always create fresh admission."""
    def validate_compiler_inputs():
        REVIEW.render_review_outputs(verified, check=True)
        for name, raw in verified.auxiliary_expected:
            if REVIEW.safe_bytes(REPO, name) != raw:
                raise ValueError('figure/README output differs from expected rendering')
    validate_compiler_inputs()
    root = REPO / 'target/paper-build'
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SOURCE_DATE_EPOCH=os.environ.get('SOURCE_DATE_EPOCH', '1760000000'))
    # Children consume captured bytes in a private input tree, never live
    # manuscript or macro paths after a parent-only fingerprint check.
    snapshot_repo = root / 'inputs' / receipt.directory.name
    snapshot_paper = snapshot_repo / 'docs/whitepaper'
    with (REPO / 'target/paper-latex.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        validate_compiler_inputs()
        for name, raw in (verified.inputs + verified.code + verified.consumed
                          + verified.expected + verified.auxiliary_expected):
            path = snapshot_repo / name
            if path.resolve() != snapshot_repo.resolve() / name:
                raise ValueError('redirected compiler snapshot')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        # Refresh only this build's admitted inventory; stale source files in a
        # reused snapshot are refused rather than silently fed to the children.
        admitted = {name for name, _ in (verified.inputs + verified.code + verified.consumed
                                        + verified.expected + verified.auxiliary_expected)}
        derived = {'docs/whitepaper/' + stem + '.pdf' for stem in PDF_STEMS}
        derived.update({'docs/whitepaper/' + stem + '.tex' for stem in PDF_STEMS if stem.endswith('-anonymous')})
        derived.update({'docs/whitepaper/figures/architecture.pdf',
                        'docs/whitepaper/generated/architecture-figure-hashes.json'})
        unexpected = {p.relative_to(snapshot_repo).as_posix() for p in snapshot_repo.rglob('*')
                      if p.is_file() and 'target' not in p.relative_to(snapshot_repo).parts}
        if unexpected - admitted - derived:
            raise ValueError('stale compiler snapshot inventory')
        run(['nice', '-n', '10', 'ionice', '-c', '3', 'pdflatex', f'-output-directory={root}',
             '-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
             '../../ops/ci/paper-architecture.tex'], receipt, cwd=snapshot_paper, env=env)
        validate_compiler_inputs()
        shutil.copyfile(root / 'paper-architecture.pdf', snapshot_paper / 'figures/architecture.pdf')
        sources = ('figures/architecture.tex', 'generated/numbers.tex', '../../ops/ci/paper-architecture.tex')
        payload = {'format': 'dope-paper-architecture-hashes-v1',
                   'source_sha256': {name: hashlib.sha256((snapshot_paper / name).read_bytes()).hexdigest() for name in sources},
                   'pdf_sha256': {'architecture.pdf': hashlib.sha256((snapshot_paper / 'figures/architecture.pdf').read_bytes()).hexdigest()}}
        (snapshot_paper / 'generated/architecture-figure-hashes.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        anonymous_sources(snapshot_paper, paper=snapshot_paper)
        for stem in PDF_STEMS:
            validate_compiler_inputs()
            run(['nice', '-n', '10', 'ionice', '-c', '3', 'latexmk', '-g', '-pdf',
                 '-usepretex=\\pdftrailerid{}\\pdfsuppressptexinfo=-1', f'-outdir={root}',
                 '-interaction=nonstopmode', '-halt-on-error', '-file-line-error', stem],
                receipt, cwd=snapshot_paper, env=env)
            validate_compiler_inputs()
            shutil.copyfile(root / f'{stem}.log', root / f'{stem}-3.log')
            shutil.copyfile(root / f'{stem}.pdf', snapshot_paper / f'{stem}.pdf')
        logs = snapshot_repo / 'target/paper-build'
        logs.mkdir(parents=True, exist_ok=True)
        for name in ('paper-architecture.log', *(stem + '-3.log' for stem in PDF_STEMS)):
            shutil.copyfile(root / name, logs / name)
        for stem in PDF_STEMS:
            run(['pdfinfo', f'{stem}.pdf'], receipt, cwd=snapshot_paper)
        validate_compiler_inputs()
        # The original paper checks also run from captured code/data. No public
        # compiler output is copied until all four PDFs and these checks pass.
        run([sys.executable, '-B', str(snapshot_paper / 'scripts/check_paper.py')],
            receipt, cwd=snapshot_repo)
        validate_compiler_inputs()
        for name in COMPILER_OUTPUTS:
            shutil.copyfile(snapshot_repo / name, REPO / name)
    return snapshot_repo


def build_paper(*, check=False, pdf_only=False):
    receipt = BuildReceipt('check' if check else ('pdf' if pdf_only else 'build'))
    verified = receipt.stage('authenticate-and-expected-render',
                             lambda: REVIEW.authenticate_build(log_directory=receipt.directory))
    receipt.bind(verified)
    receipt.stage('review-output-validation' if check or pdf_only else 'review-output-publication',
                  lambda: REVIEW.render_review_outputs(verified, check=check or pdf_only))
    verified = receipt.stage('figures-and-readme', lambda: auxiliary_outputs(
        verified, check=check or pdf_only, receipt=receipt))
    receipt.bind(verified)
    # Includes every legacy/review output; checks never repair drift by writing.
    receipt.stage('precompile-zero-drift', lambda: output_drift(receipt, before_compile=True))
    receipt.stage('all-four-pdfs', lambda: compile_pdfs(verified, receipt))
    receipt.stage('postcompile-zero-drift', lambda: output_drift(receipt))
    def final_identity():
        verified.revalidate()
        REVIEW.render_review_outputs(verified, check=True)
        receipt.payload['compiled_pdf_sha256'] = {stem: hashlib.sha256((PAPER / f'{stem}.pdf').read_bytes()).hexdigest()
                                                  for stem in PDF_STEMS}
    receipt.stage('final-context-and-pdf-identities', final_identity)
    receipt.payload['status'] = 'passed'
    receipt.save()
    return verified


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--pdf-only', action='store_true')
    args = parser.parse_args()
    build_paper(check=args.check, pdf_only=args.pdf_only)


if __name__ == '__main__':
    main()
