#!/usr/bin/env python3
"""hostload.py -- sample the host's CPU load beside a running server, one JSON line per sample.

    hostload.py <out.jsonl> [period_s (2)] [cgroup dir of the server]

Each line: t, the load averages, the cores busy over the last period on the whole host (/proc/stat),
and -- when a cgroup is given -- the cores the server's cgroup used (cpu.stat usage_usec), the rest
("foreign": everything that is not the server, the harness's own clients included), the three
foreign processes that used the most CPU (so a disturbance can be named, not only sized), the whole
host's disk read/write MB/s (/proc/diskstats, whole disks), and how many processes OUTSIDE the
server's cgroup hold the GPU (/sys/class/kfd/kfd/proc: a squatter, or another job that started
without the lock). The load average alone cannot tell a busy server from a busy neighbour: the
server's own threads are part of it. Runs until killed, or until the server's cgroup
is gone. Linux only (procfs, cgroup v2, amdkfd). radiance-exl3 P0. Apache-2.0.
"""
import json
import os
import sys
import time

HZ = os.sysconf("SC_CLK_TCK")


def host_busy_s():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle = v[3] + v[4]                      # idle + iowait
    return (sum(v[:8]) - idle) / HZ


def cgroup_s(cg):
    with open(os.path.join(cg, "cpu.stat")) as f:
        for line in f:
            k, val = line.split()
            if k == "usage_usec":
                return int(val) / 1e6
    raise RuntimeError(f"no usage_usec in {cg}/cpu.stat")


def proc_ticks():
    """{pid: (name, utime + stime ticks)} for every process and kernel thread."""
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as f:
                s = f.read()
        except OSError:
            continue
        rest = s[s.rindex(")") + 2:].split()
        out[pid] = (s[s.index("(") + 1:s.rindex(")")], int(rest[11]) + int(rest[12]))
    return out


def in_cgroup(pid, cg_rel):
    try:
        with open(f"/proc/{pid}/cgroup") as f:
            return f.read().strip() == f"0::{cg_rel}"
    except OSError:
        return False


def top_foreign(before, after, dt, cg_rel, n=3):
    """The n processes outside the server's cgroup that used the most CPU between two proc_ticks()."""
    used = sorted(((after[p][1] - before[p][1], p) for p in after if p in before), reverse=True)
    top = []
    for d, p in used:
        if d <= 0 or len(top) == n:
            break
        if not in_cgroup(p, cg_rel):
            top.append([after[p][0], round(d / HZ / dt, 2)])
    return top


def disk_mb():
    """Bytes read and written so far by whole disks (nvme*n*, sd*, vd*), in MB."""
    r = w = 0
    with open("/proc/diskstats") as f:
        for line in f:
            v = line.split()
            name = v[2]
            whole = (name.startswith("nvme") and "p" not in name[4:]) or \
                    (name[:2] in ("sd", "vd") and not name[-1].isdigit())
            if whole:
                r, w = r + int(v[5]), w + int(v[9])
    return r * 512 / 1e6, w * 512 / 1e6


def kfd_foreign(cg_rel):
    """Processes holding /dev/kfd whose cgroup is not the server's."""
    n = 0
    for pid in os.listdir("/sys/class/kfd/kfd/proc"):
        try:
            with open(f"/proc/{pid}/cgroup") as f:
                mine = f.read().strip() == f"0::{cg_rel}"
        except OSError:                     # gone, or not ours to read: a GPU client all the same
            mine = False
        n += not mine
    return n


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: hostload.py <out.jsonl> [period_s] [cgroup dir]")
    out = sys.argv[1]
    period = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    cg = sys.argv[3] if len(sys.argv) > 3 else None
    t0, b0, d0 = time.time(), host_busy_s(), disk_mb()
    s0 = cgroup_s(cg) if cg else None
    p0 = proc_ticks() if cg else None
    with open(out, "a") as f:
        while True:
            time.sleep(period)
            t1, b1, d1 = time.time(), host_busy_s(), disk_mb()
            with open("/proc/loadavg") as la:
                load = [float(x) for x in la.read().split()[:3]]
            row = {"t": round(t1, 3), "load1": load[0], "load5": load[1], "load15": load[2],
                   "busy_cores": round((b1 - b0) / (t1 - t0), 3),
                   "disk_read_mb_s": round((d1[0] - d0[0]) / (t1 - t0), 1),
                   "disk_write_mb_s": round((d1[1] - d0[1]) / (t1 - t0), 1)}
            if cg:
                try:
                    s1 = cgroup_s(cg)
                except OSError:              # the server's cgroup is gone: it stopped
                    break
                row["server_cores"] = round((s1 - s0) / (t1 - t0), 3)
                row["foreign_cores"] = round(row["busy_cores"] - row["server_cores"], 3)
                row["kfd_foreign"] = kfd_foreign(cg[len("/sys/fs/cgroup"):])
                p1 = proc_ticks()
                row["top_foreign"] = top_foreign(p0, p1, t1 - t0, cg[len("/sys/fs/cgroup"):])
                s0, p0 = s1, p1
            f.write(json.dumps(row) + "\n")
            f.flush()
            t0, b0, d0 = t1, b1, d1


if __name__ == "__main__":
    main()
