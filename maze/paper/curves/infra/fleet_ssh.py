#!/usr/bin/env python3
"""Pure-Python SSH helper for RunPod pods (paramiko), for sessions where the system `ssh` binary cannot run
(e.g. a background job that lost access to macOS directory services: "No user exists for uid 501", no DNS).
Talks to pods by IP:port only. Needs paramiko on PYTHONPATH.

  python fleet_ssh.py run   <ip:port> '<shell command>'
  python fleet_ssh.py put   <ip:port> <remote_dir> <local_base> <path> [<path> ...]   (tar-streams paths relative to local_base)
  python fleet_ssh.py putlist <ip:port> <remote_dir> <local_base> <file_with_relative_paths>
  python fleet_ssh.py get   <ip:port> <remote_path> <local_dir> [--exclude GLOB ...]    (recursive sftp download)
  python fleet_ssh.py wait  <ip:port>                                                   (until sshd answers)
"""
import fnmatch, io, os, stat, sys, tarfile, threading, time
import paramiko

KEY = os.path.expanduser(os.environ.get("RUNPOD_SSH_KEY", "~/.ssh/runpod"))
ENV = "[ -f /etc/rp_environment ] && . /etc/rp_environment; "


def connect(ep, timeout=20):
    host, port = ep.rsplit(":", 1)
    c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=int(port), username="root", pkey=paramiko.Ed25519Key.from_private_key_file(KEY), timeout=timeout,
              banner_timeout=timeout, auth_timeout=timeout, allow_agent=False, look_for_keys=False)
    c.get_transport().set_keepalive(30)
    return c


def run(ep, cmd, timeout=None, stdin_data=None):
    c = connect(ep)
    try:
        i, o, e = c.exec_command(ENV + cmd, timeout=timeout)
        if stdin_data is not None: i.write(stdin_data)
        i.channel.shutdown_write()
        out = o.read().decode(errors="replace"); err = e.read().decode(errors="replace"); rc = o.channel.recv_exit_status()
        return rc, out, err
    finally:
        c.close()


def put(ep, remote_dir, local_base, paths, excludes=("__pycache__", ".git", "*.pyc", ".DS_Store")):
    """Stream a gzip tar of `paths` (relative to local_base) into remote_dir."""
    c = connect(ep)
    try:
        ch = c.get_transport().open_session(); ch.exec_command(f"mkdir -p {remote_dir} && tar -C {remote_dir} --no-same-owner --no-same-permissions -xzf -")
        f = ch.makefile("wb", 1 << 20)
        def flt(ti):
            base = os.path.basename(ti.name)
            return None if any(fnmatch.fnmatch(base, x) for x in excludes) else ti
        with tarfile.open(fileobj=f, mode="w|gz", compresslevel=1) as tf:
            for p in paths:
                tf.add(os.path.join(local_base, p), arcname=p, filter=flt)
        f.flush(); ch.shutdown_write(); rc = ch.recv_exit_status()
        if rc != 0: raise RuntimeError(f"remote tar failed rc={rc}: {ch.recv_stderr(4000).decode(errors='replace')}")
    finally:
        c.close()


def get(ep, remote_path, local_dir, excludes=()):
    c = connect(ep); n = 0
    try:
        sftp = c.open_sftp()
        def walk(r, l):
            nonlocal n
            os.makedirs(l, exist_ok=True)
            for a in sftp.listdir_attr(r):
                if any(fnmatch.fnmatch(a.filename, x) for x in excludes): continue
                rp, lp = f"{r}/{a.filename}", os.path.join(l, a.filename)
                if stat.S_ISDIR(a.st_mode): walk(rp, lp)
                elif not (os.path.exists(lp) and os.path.getsize(lp) == a.st_size):
                    sftp.get(rp, lp); n += 1
        st = sftp.stat(remote_path)
        if stat.S_ISDIR(st.st_mode): walk(remote_path.rstrip("/"), local_dir)
        else:
            os.makedirs(local_dir, exist_ok=True); sftp.get(remote_path, os.path.join(local_dir, os.path.basename(remote_path))); n += 1
    finally:
        c.close()
    return n


def wait(ep, tries=60):
    for _ in range(tries):
        try:
            rc, out, _ = run(ep, "echo ready", timeout=15)
            if "ready" in out: return True
        except Exception:
            pass
        time.sleep(5)
    return False


if __name__ == "__main__":
    act, ep = sys.argv[1], sys.argv[2]
    if act == "run":
        rc, out, err = run(ep, sys.argv[3]); sys.stdout.write(out); sys.stderr.write(err); sys.exit(rc)
    elif act == "put":
        put(ep, sys.argv[3], sys.argv[4], sys.argv[5:])
    elif act == "putlist":
        put(ep, sys.argv[3], sys.argv[4], [l.strip() for l in open(sys.argv[5]) if l.strip()])
    elif act == "get":
        ex = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--exclude"]
        print(get(ep, sys.argv[3], sys.argv[4], ex), "files")
    elif act == "wait":
        sys.exit(0 if wait(ep) else 1)
