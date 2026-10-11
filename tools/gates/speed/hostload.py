#!/usr/bin/env python3
"""hostload.py -- sample the host's CPU load beside a running server, one JSON line per sample.

    hostload.py <out.jsonl> [period_s (2)] [cgroup dir of the server]

Each line: t, the load averages, the cores busy over the last period on the whole host (/proc/stat),
and -- when a cgroup is given -- the cores the server's cgroup used (cpu.stat usage_usec), the rest
("foreign": everything that is not the server, the harness's own clients included), and how many
processes OUTSIDE the server's cgroup hold the GPU (/sys/class/kfd/kfd/proc: a squatter, or another
job that started without the lock). The load average alone cannot tell a busy server from a busy
neighbour: the server's own threads are part of it. Runs until killed, or until the server's cgroup
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
    t0, b0 = time.time(), host_busy_s()
    s0 = cgroup_s(cg) if cg else None
    with open(out, "a") as f:
        while True:
            time.sleep(period)
            t1, b1 = time.time(), host_busy_s()
            with open("/proc/loadavg") as la:
                load = [float(x) for x in la.read().split()[:3]]
            row = {"t": round(t1, 3), "load1": load[0], "load5": load[1], "load15": load[2],
                   "busy_cores": round((b1 - b0) / (t1 - t0), 3)}
            if cg:
                try:
                    s1 = cgroup_s(cg)
                except OSError:              # the server's cgroup is gone: it stopped
                    break
                row["server_cores"] = round((s1 - s0) / (t1 - t0), 3)
                row["foreign_cores"] = round(row["busy_cores"] - row["server_cores"], 3)
                row["kfd_foreign"] = kfd_foreign(cg[len("/sys/fs/cgroup"):])
                s0 = s1
            f.write(json.dumps(row) + "\n")
            f.flush()
            t0, b0 = t1, b1


if __name__ == "__main__":
    main()
