"""Metadata-only pidfd capture controls; no scientific initialization."""
import os
from pathlib import Path
import select
import subprocess
import sys
import unittest
from unittest.mock import patch
from research.benchmark import tabsyn_owned_lifecycle as lifecycle


@unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'pidfd_open'), 'Linux pidfd controls')
class OwnedCapture(unittest.TestCase):
    def spawn(self):
        r,w=os.pipe()
        child=subprocess.Popen([sys.executable,'-I','-S','-B','-c',
            'import os,sys; os.read(int(sys.argv[1]),1)',str(r)],pass_fds=(r,),
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            start_new_session=True)
        os.close(r)
        def cleanup():
            try: os.close(w)
            except OSError: pass
            child.wait(timeout=5)
        self.addCleanup(cleanup)
        return child,w

    def finish(self,child,w):
        os.write(w,b'x');os.close(w)
        self.assertEqual(child.wait(timeout=5),0)
        self.assertFalse(Path('/proc',str(child.pid)).exists())

    def test_actual_live_capture(self):
        child,w=self.spawn();backend=lifecycle.Linux();row=backend.table()[child.pid]
        fd=backend.handle(row)
        self.assertIsInstance(fd,int);self.assertFalse(backend.exited(fd))
        self.finish(child,w);self.assertTrue(backend.exited(fd));backend.close_handle(fd)
        with self.assertRaises(OSError):os.fstat(fd)

    def test_actual_concurrent_reap_is_unbound_and_not_owned(self):
        child,w=self.spawn();backend=lifecycle.Linux();row=backend.table()[child.pid]
        original=backend.table;tables=[{child.pid:row}];fds=[];events=[]
        backend.table=lambda:tables.pop(0) if tables else original()
        def opened(pid):
            fd=os.pidfd_open(pid);fds.append(fd);self.finish(child,w)
            self.assertTrue(select.select([fd],[],[],0)[0]);return fd
        backend._pidfd_api=(opened,None)
        guardian=lifecycle.Guardian(backend,lambda k,f:events.append((k,f)))
        with patch.object(backend,'signal',side_effect=AssertionError('signal forbidden')):
            self.assertEqual(guardian.capture(),{});guardian.close()
        self.assertFalse(guardian.known)
        self.assertTrue(any(k=='capture_subject_vanished_unbound' for k,f in events))
        self.assertFalse(any(k=='owned_descendant_captured' for k,f in events))
        with self.assertRaises(OSError):os.fstat(fds[0])

    def test_absence_requires_dead_pidfd_and_identity_drift_rejects(self):
        row=lifecycle.Linux().table()[os.getpid()]
        for label,again,exited,absent in [
            ('live_pidfd',None,False,True),('remaining_PID',None,True,False),
            ('UID',dict(row,uid=row['uid']+1),True,True),
            ('tick',dict(row,start_ticks=row['start_ticks']+1),True,True),
            ('group',dict(row,group=row['group']+1),True,True)]:
            with self.subTest(label=label):
                backend=lifecycle.Linux();fd=os.pidfd_open(os.getpid())
                backend._pidfd_api=(lambda pid:fd,None)
                backend.table=lambda:{row['pid']:again} if again is not None else {}
                backend.exited=lambda handle:exited;backend.absent=lambda pid:absent
                with self.assertRaises(AssertionError):backend.handle(row)
                with self.assertRaises(OSError):os.fstat(fd)
        backend=lifecycle.Linux()
        with patch.object(backend,'prepare_pidfd',side_effect=AssertionError('open forbidden')):
            with self.assertRaisesRegex(AssertionError,'UID'):
                backend.handle(dict(row,uid=row['uid']+1))
        backend._pidfd_api=(lambda pid:(_ for _ in ()).throw(ProcessLookupError()),None)
        with self.assertRaises(ProcessLookupError):backend.handle(row)
