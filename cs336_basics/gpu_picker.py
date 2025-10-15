# gpu_picker.py
import os
import time
import fcntl
from typing import Tuple, Optional

def _query_gpus_with_nvidia_smi():
    import subprocess
    # device ids
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
    )
    ids = [int(x.strip()) for x in out.strip().splitlines() if x.strip() != ""]

    # used memory (MB)
    out2 = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    used = [int(x.strip()) for x in out2.strip().splitlines() if x.strip() != ""]
    return ids, used

def _query_gpus():
    return _query_gpus_with_nvidia_smi()

def _try_lock(gpu_id: int) -> Optional[object]:
    """
    Try to acquire a non-blocking file lock for this GPU.
    Returns an open file handle if successful (keep it alive!), else None.
    """
    path = f"/tmp/gpu_{gpu_id}.lock"
    f = open(path, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except BlockingIOError:
        f.close()
        return None

def get_gpu(threshold_mb: int = 100,
            sleep_seconds: int = 600,
            start_index: int = 0) -> Tuple[int, object]:
    """
    Returns (gpu_id, lock_handle).
    - Skips GPUs whose used memory > threshold_mb.
    - If none are free, sleeps 'sleep_seconds' and retries.
    - Round-robin starting point advances each cycle to avoid starvation.
    Keep the returned lock_handle open for the lifetime of your process.
    """
    # Make CUDA pick the “visible” device ordering by PCI bus id
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    while True:
        gpu_ids, used_list = _query_gpus()
        if not gpu_ids:
            raise RuntimeError("No GPUs detected (nvidia-smi / NVML unavailable).")

        n = len(gpu_ids)
        # Round-robin order for this cycle
        order = [(start_index + k) % n for k in range(n)]

        for idx in order:
            gid = gpu_ids[idx]
            used_mb = used_list[idx]
            if used_mb <= threshold_mb:
                lock = _try_lock(gid)
                if lock is None:
                    # Someone else grabbed it between check & lock—try next
                    continue
                # Success: pin this process to that GPU
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)
                return gid, lock

        # None available: wait and try again, starting from next GPU (round-robin)
        print(f"Sleeping for {sleep_seconds}")
        time.sleep(sleep_seconds)
        start_index = (start_index + 1) % len(gpu_ids)
