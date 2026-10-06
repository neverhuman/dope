"""Linux subreaper plus pidfd custody; lease closes after descendants, never before."""
import os
from pathlib import Path
import select
import signal
import subprocess
import time


# Distinct capture outcome: no owned identity, GPU charge, or signal authority.
UNBOUND_VANISHED = object()


class Linux:
    def __init__(self, native_authorized=False):
        self.authorized = native_authorized; self.children = {}; self._pidfd_api = None

    def prepare_pidfd(self):
        if self._pidfd_api is not None: return self._pidfd_api
        if callable(getattr(os, 'pidfd_open', None)) and callable(getattr(signal, 'pidfd_send_signal', None)):
            self._pidfd_api = (os.pidfd_open, signal.pidfd_send_signal)
            return self._pidfd_api
        # Header-bound Linux x86_64 syscall numbers; never a PID signal fallback.
        # The caller verifies the original runtime/providers before this import.
        if os.uname().sysname != 'Linux' or os.uname().machine != 'x86_64':
            raise RuntimeError('pidfd_fallback_architecture_unadmitted')
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        syscall = libc.syscall
        syscall.argtypes = [ctypes.c_long]  # Variadic tail is explicitly typed below.
        syscall.restype = ctypes.c_long
        def opened(pid):
            assert type(pid) is int and 0 < pid <= 2147483647
            ctypes.set_errno(0)
            fd = int(syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0)))
            if fd < 0:
                code = ctypes.get_errno()
                if not code: raise RuntimeError('owned_pidfd_open_failed_without_errno')
                raise OSError(code, 'owned_pidfd_open_failed')
            assert fd <= 2147483647
            return fd
        def sent(fd, signum):
            assert type(fd) is int and fd >= 0 and type(signum) is int
            ctypes.set_errno(0)
            status = int(syscall(ctypes.c_long(424), ctypes.c_int(fd), ctypes.c_int(signum),
                ctypes.c_void_p(None), ctypes.c_uint(0)))
            if status < 0:
                code = ctypes.get_errno()
                if not code: raise RuntimeError('owned_pidfd_signal_failed_without_errno')
                raise OSError(code, 'owned_pidfd_signal_failed')
            assert status == 0
        self._pidfd_api = (opened, sent)
        return self._pidfd_api

    def enable(self):
        assert self.authorized
        self.prepare_pidfd()
        # Called only after prefrozen candidate/driver/tool bytes and root grant
        # pass. libc is already a pinned loader provider, no new DSO path here.
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        libc.prctl.restype = ctypes.c_int
        ctypes.set_errno(0)
        if libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'owned_subreaper_enable_failed')
        assert signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL

    def table(self):
        result = {}
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                text = path.read_text(); row = text[text.rfind(')') + 2:].split(); pid = int(path.parent.name)
                result[pid] = {'pid': pid, 'parent_pid': int(row[1]), 'uid': path.parent.stat().st_uid,
                    'group': int(row[2]), 'start_ticks': int(row[19]), 'state': row[0]}
            except (FileNotFoundError, ProcessLookupError): continue
        return result

    def handle(self, row):
        assert row['uid'] == self.uid(), 'pidfd subject UID differs'
        fd = self.prepare_pidfd()[0](row['pid'])
        try:
            again = self.table().get(row['pid'])
            if again is None:
                # A successfully opened pidfd anchors this one historical
                # subject. Absence alone never permits an unbound capture.
                assert self.exited(fd), 'post_open_absent_subject_pidfd_not_exited'
                assert self.absent(row['pid']), 'post_open_absent_subject_PID_present'
                os.close(fd)
                return UNBOUND_VANISHED
            assert identity(again) == identity(row)
        except BaseException:
            os.close(fd)
            raise
        return fd

    def signal(self, handle, signum):
        assert self.authorized and signum in (signal.SIGTERM, signal.SIGKILL)
        return self.prepare_pidfd()[1](handle, int(signum))
    def exited(self, handle): return bool(select.select([handle], [], [], 0)[0])
    def absent(self, pid): return not Path('/proc', str(pid)).exists()
    def close_handle(self, handle): os.close(handle)
    def sleep(self, seconds): time.sleep(seconds)
    def now(self): return time.monotonic()
    def parent_pid(self): return os.getpid()
    def uid(self): return os.getuid()

    def spawn(self, argv, env, cwd, log):
        assert self.authorized
        child = subprocess.Popen(argv, env=env, cwd=cwd, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        self.children[child.pid] = child
        return child

    def reap(self, pid):
        if pid in self.children:
            self.children[pid].poll()
        else:
            try: os.waitpid(pid, os.WNOHANG)
            except ChildProcessError: pass


def identity(row): return (row['uid'], row['group'], row['start_ticks'])


class Guardian:
    def __init__(self, backend, event):
        self.backend = backend; self.event = event; self.known = {}

    def capture(self):
        # Subreaper ancestry preserves children even when the original leader
        # exited before the first poll. Capture precedes every reap/poll.
        rows = self.backend.table(); root = self.backend.parent_pid()
        pending = dict(rows); progress = True
        while progress:
            progress = False
            for pid, row in list(pending.items()):
                if pid == root: del pending[pid]; continue
                key = (pid, row['start_ticks'])
                parent = rows.get(row['parent_pid'])
                parent_owned = parent is not None and (parent['pid'], parent['start_ticks']) in self.known
                if key not in self.known and row['parent_pid'] != root and not parent_owned: continue
                assert row['uid'] == self.backend.uid(), 'owned descendant UID changed'
                if key not in self.known:
                    handle = self.backend.handle(row)
                    if handle is UNBOUND_VANISHED:
                        # No handle or identity is retained for signaling,
                        # stale GPU-row explanation, or ownership accounting.
                        self.event('capture_subject_vanished_unbound', {
                            'snapshot_pid': pid,
                            'snapshot_uid': row['uid'],
                            'snapshot_start_ticks': row['start_ticks'],
                            'snapshot_parent_pid': row['parent_pid'],
                            'added_to_owned_known': False})
                        del pending[pid]; progress = True
                        continue
                    self.known[key] = {'row': row, 'handle': handle}
                    self.event('owned_descendant_captured', {'pid': pid, 'identity': identity(row),
                        'parent_pid': row['parent_pid']})
                else:
                    # pidfd anchors PID ownership; UID/start-tick changes reject.
                    old = self.known[key]['row']
                    assert row['uid'] == old['uid'] and row['start_ticks'] == old['start_ticks']
                del pending[pid]; progress = True
        return self.known

    def active(self):
        self.capture()
        return {key: value for key, value in self.known.items() if not self.backend.exited(value['handle'])}

    def close(self):
        self.capture(); started = self.backend.now(); term_sent = set(); killed = set()
        while True:
            self.capture()
            for key, value in self.known.items():
                pid = value['row']['pid']
                if self.backend.exited(value['handle']): self.backend.reap(pid); continue
                if key not in term_sent:
                    self.backend.signal(value['handle'], signal.SIGTERM); term_sent.add(key)
                if self.backend.now() - started >= 2 and key not in killed:
                    self.backend.signal(value['handle'], signal.SIGKILL); killed.add(key)
            self.capture()
            if all(self.backend.exited(v['handle']) for v in self.known.values()):
                for value in self.known.values(): self.backend.reap(value['row']['pid'])
                # One final adoption scan closes the immediate-exit race.
                before = len(self.known); self.capture()
                if len(self.known) == before and not self.active(): break
            self.event('descendant_cleanup_pending', {'elapsed_seconds': self.backend.now() - started})
            self.backend.sleep(.05)
        for value in self.known.values(): self.backend.close_handle(value['handle'])
        self.event('descendant_lease_closed', {'identities': {str(key): identity(v['row']) for key, v in self.known.items()},
            'cleanup_seconds': self.backend.now() - started})
