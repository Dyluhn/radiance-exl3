#!/usr/bin/env python3
"""clocks.py -- sample one card's clocks while a calibration session runs, one JSON line per sample.

    clocks.py <pci address> <out.jsonl> [period_s]

Source, named in every line: the LACT daemon socket (CALIB_LACT_SOCK, default /run/lactd.sock; no root
needed when the socket is group-readable) when it answers for this card, else the card's sysfs
pp_dpm_sclk / pp_dpm_mclk (the starred level, which is a DPM level and not a measured clock). Runs until
killed. radiance-exl3 S0.11. Apache-2.0.
"""
import json
import os
import socket
import sys
import time


def lact(sock_path, command, args=None):
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(2.0)
    s.connect(sock_path)
    msg = {"command": command} if args is None else {"command": command, "args": args}
    s.sendall((json.dumps(msg) + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(1 << 20)
        if not chunk:
            break
        buf += chunk
    s.close()
    reply = json.loads(buf)
    if reply.get("status") != "ok":
        raise RuntimeError(f"LACT {command}: {reply}")
    return reply["data"]


def lact_device_id(sock_path, pci):
    for dev in lact(sock_path, "list_devices"):
        if dev["id"].lower().endswith(pci.lower()):
            return dev["id"]
    raise RuntimeError(f"LACT lists no device at {pci}")


def lact_sample(sock_path, dev_id):
    st = lact(sock_path, "device_stats", {"id": dev_id})
    return {"src": "lact", "sclk": st["clockspeed"]["gpu_clockspeed"], "mclk": st["clockspeed"]["vram_clockspeed"],
            "busy": st["busy_percent"], "power_w": st["power"]["average"], "cap_w": st["power"]["cap_current"],
            "tj_c": st["temps"]["junction"]["current"], "throttle": st["throttle_info"],
            "level": st["performance_level"]}


def starred(path):
    with open(path) as f:
        for line in f:
            if line.rstrip().endswith("*"):
                return int(line.split(":")[1].strip().rstrip("*").strip().lower().rstrip("mhz"))
    return None


def sysfs_sample(dev_dir):
    return {"src": "sysfs-dpm-level", "sclk": starred(os.path.join(dev_dir, "pp_dpm_sclk")),
            "mclk": starred(os.path.join(dev_dir, "pp_dpm_mclk"))}


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: clocks.py <pci address> <out.jsonl> [period_s]")
    pci, out = sys.argv[1], sys.argv[2]
    period = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    sock_path = os.environ.get("CALIB_LACT_SOCK", "/run/lactd.sock")
    dev_dir = f"/sys/bus/pci/devices/{pci}"
    try:
        dev_id = lact_device_id(sock_path, pci)
        sample = lambda: lact_sample(sock_path, dev_id)
        sample()
    except (OSError, RuntimeError, ValueError, KeyError) as e:
        print(f"clocks.py: LACT unavailable ({e}); falling back to sysfs DPM levels", file=sys.stderr)
        sample = lambda: sysfs_sample(dev_dir)
    with open(out, "a") as f:
        while True:
            row = {"t": round(time.time(), 3)}
            row.update(sample())
            f.write(json.dumps(row) + "\n")
            f.flush()
            time.sleep(period)


if __name__ == "__main__":
    main()
