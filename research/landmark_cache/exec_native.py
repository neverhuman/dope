"""Apply limits after exec, avoiding preexec_fn in a multithreaded parent."""
import os
import resource
import sys

if __name__ == "__main__":
    memory, cpu = map(int, sys.argv[1:3])
    os.umask(0o077)
    os.nice(10)
    # Darwin rejects finite RLIMIT_AS even when getrlimit reports infinity.
    # Its parent uses an explicitly reported sampled-RSS watchdog instead.
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    os.execv(sys.argv[3], sys.argv[3:])
