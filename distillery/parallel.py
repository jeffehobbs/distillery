"""Work out how many workers this machine can actually run, then run them.

The point is to scale to whatever hardware this lands on rather than hardcoding a
number for one 18-core Mac. Core count is resolved in this order:

  1. `DISTILLERY_WORKERS` (or a `--workers` flag) if set — an explicit override wins.
  2. `os.process_cpu_count()` (3.13+), which already accounts for CPU affinity.
  3. `len(os.sched_getaffinity(0))` on Linux — respects taskset/cpuset pinning,
     which `os.cpu_count()` does not.
  4. `os.cpu_count()`.

…then clamped by the **cgroup CPU quota** if there is one, so a 2-CPU container on
a 64-core host uses 2 workers and not 64; and clamped again by RAM for stages where
each worker holds a whole track (or a 9-minute buffer) in memory, since on a small
machine the memory ceiling binds long before the core count does.

Every stage here is *task*-parallel over independent items, so results are collected
in submission order and reductions happen on one thread. That keeps a run
bit-identical regardless of how many workers it used — the seed still reproduces the
mix exactly.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ENV_WORKERS = "DISTILLERY_WORKERS"
_OVERRIDE = None          # set by --workers


def set_override(n):
    """CLI override; None or 0 means auto."""
    global _OVERRIDE
    _OVERRIDE = int(n) if n else None


def _cgroup_quota():
    """CPU limit from cgroup v2 or v1, in whole CPUs, or None."""
    try:                                          # cgroup v2: "max 100000" | "200000 100000"
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = (fh.read().split() + ["100000"])[:2]
        if quota != "max":
            return max(1, int(float(quota) / float(period)))
    except (OSError, ValueError):
        pass
    try:                                          # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            quota = int(fh.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            period = int(fh.read().strip())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return None


def available_cpus():
    """Usable CPUs on this machine, honouring affinity and container quotas."""
    n = None
    if hasattr(os, "process_cpu_count"):          # Python 3.13+
        n = os.process_cpu_count()
    if n is None and hasattr(os, "sched_getaffinity"):
        try:
            n = len(os.sched_getaffinity(0))
        except OSError:
            n = None
    if n is None:
        n = os.cpu_count()
    n = max(1, int(n or 1))
    q = _cgroup_quota()
    if q:
        n = max(1, min(n, q))
    return n


def total_ram_gb():
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        return 8.0


def worker_count(kind="cpu", n_items=None, per_worker_gb=0.0, reserve=0):
    """How many workers to use for a stage.

    kind='cpu'  — one worker per usable core (minus `reserve`).
    kind='io'   — oversubscribe, since these workers sit blocked on the network.
    per_worker_gb — memory each worker needs; caps workers on small machines.
    """
    explicit = _OVERRIDE
    if explicit is None:
        env = os.environ.get(ENV_WORKERS, "").strip()
        if env.isdigit() and int(env) > 0:
            explicit = int(env)

    if explicit is not None:
        n = explicit
    else:
        cpus = available_cpus()
        if kind == "io":
            n = min(max(4, cpus * 2), 16)         # network latency, not CPU
        else:
            n = max(1, cpus - max(0, reserve))
        if per_worker_gb > 0:
            # leave ~30% of RAM for the OS and the parent's own buffers
            budget = max(1, int((total_ram_gb() * 0.7) / per_worker_gb))
            n = min(n, budget)
    if n_items:
        n = min(n, max(1, int(n_items)))
    return max(1, int(n))


def describe():
    return (f"{available_cpus()} usable cores, {total_ram_gb():.0f} GB RAM"
            + (f", workers forced to {_OVERRIDE}" if _OVERRIDE else ""))


def run(fn, items, workers, on_result=None):
    """Map fn over items with `workers` threads; results come back IN ORDER.

    Threads are the right tool here even for CPU-heavy work, because each of these
    stages either shells out to a subprocess (Essentia analysis, loop extraction —
    which also keeps a segfaulting child from taking the run down with it) or spends
    its time inside numpy/scipy/pedalboard, all of which release the GIL.
    """
    if workers <= 1 or len(items) <= 1:
        out = []
        for i, item in enumerate(items):
            r = fn(item)
            out.append(r)
            if on_result:
                on_result(i, item, r)
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fn, item) for item in items]
        out = []
        for i, (item, fut) in enumerate(zip(items, futures)):
            r = fut.result()
            out.append(r)
            if on_result:
                on_result(i, item, r)
        return out


def imap_ordered(fn, items, workers, prefetch=2):
    """Like `run`, but yields (item, result) as results become ready IN ORDER while
    keeping only ~`workers + prefetch` jobs in flight.

    Needed where each result is a big buffer: collecting all of a 9-minute mix's
    loop clips up front would hold well over a gigabyte at once, which is fine on a
    64 GB desktop and not fine on a small box. This keeps peak memory proportional
    to the worker count instead of the item count, and in-order delivery keeps the
    reduction (mixing) bit-identical no matter how many workers ran.
    """
    if workers <= 1:
        for item in items:
            yield item, fn(item)
        return
    from collections import deque
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pending = deque()
        it = iter(items)
        for _ in range(workers + prefetch):
            try:
                nxt = next(it)
            except StopIteration:
                break
            pending.append((nxt, ex.submit(fn, nxt)))
        while pending:
            item, fut = pending.popleft()
            result = fut.result()
            try:
                nxt = next(it)
                pending.append((nxt, ex.submit(fn, nxt)))
            except StopIteration:
                pass
            yield item, result


def run_subprocess(argv, timeout=1800):
    """Run a worker subprocess with this interpreter; returns CompletedProcess."""
    import subprocess
    return subprocess.run([sys.executable, *argv], capture_output=True, text=True,
                          timeout=timeout)
