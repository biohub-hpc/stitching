"""Side-effect-free helpers for stripe-worker GPU pinning.

This module is imported as a side effect when ProcessPoolExecutor workers
unpickle their initializer reference. It MUST NOT import cupy / CUDA, or
the CUDA context will be initialized in the child BEFORE CUDA_VISIBLE_DEVICES
is set — defeating the per-worker GPU pinning.

Keep imports minimal (stdlib only). The actual stripe processing function
lives in stitch.stitch.assemble and is imported only when the worker
receives its first task — by which point CVD has been set by the
initializer below.
"""
import os


def stripe_worker_gpu_init(gpu_queue):
    """PPE initializer: assign this worker one GPU.

    Pulls a gpu_id from the shared queue and sets CUDA_VISIBLE_DEVICES
    to that single device. Subsequent `import cupy` in this worker
    (triggered by the first task's resolution of the function in
    stitch.stitch.assemble) will see only the assigned GPU.

    On queue exhaustion or any error, falls back to GPU 0.
    """
    try:
        gpu_id = gpu_queue.get(timeout=5)
    except Exception:
        gpu_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        print(
            f"[Stripe worker pid={os.getpid()}] pinned to GPU {gpu_id} "
            f"via CUDA_VISIBLE_DEVICES (before cupy import)"
        )
    except Exception:
        pass
