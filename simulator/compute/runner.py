"""
runner.py

Author: natelgrw
Last Edited: 04/01/2026

Handles parallel execution of circuit simulations.
Separates compute logic from CLI user interface.
"""

import multiprocessing as mp
import os
import time


# ===== Global Worker State ===== #


# Worker process global to avoid serialization bottleneck
_worker_extractor = None

def _worker_init(extractor):
    """
    Initialize worker process with extractor instance.
    Called once per worker, not once per task (fixes serialization bottleneck).
    
    Parameters:
    -----------
    extractor: Configured Extractor instance.
    """
    global _worker_extractor
    _worker_extractor = extractor


# ===== Adaptive Worker Utilities ===== #


MIN_FREE_CORES = 2
MICRO_BATCH_INTERVAL = 1000
ADAPTIVE_RESIZE_HYSTERESIS = 4
PREFLIGHT_CHECK_MAX_FILES = 10000  # Skip preflight for larger result directories


def worker_task(args):
    """
    Execute a single simulation worker task using global _worker_extractor.
    This avoids serialization overhead by using process-local state.

    Parameters:
    -----------
    args (tuple): (index, config).

    Returns:
    --------
    tuple: (success, payload) where payload is (idx, value_or_error).
    """
    global _worker_extractor
    
    idx, config = args
    try:
        value = _worker_extractor(config, sim_id=idx)
        return True, (idx, value)
    except Exception as exc:
        import traceback

        err_msg = f"Sim {idx} Failed: {str(exc)}\n{traceback.format_exc()}"
        print(f"\n[WORKER ERROR] {err_msg}", flush=True)
        return False, (idx, err_msg)


def _format_worker_result(success, payload):
    """
    Normalize worker results into a common tuple shape.

    Parameters:
    -----------
    success (bool): Whether worker task succeeded.
    payload (tuple): (idx, data_or_error).

    Returns:
    --------
    tuple: (idx, extractor_like_result).
    """
    idx, data = payload
    if not success:
        print(f"\n[!] SIMULATION FAILED - ID {idx}: {data}", flush=True)
        return idx, (0.0, {'valid': False})
    return idx, data


def _run_preflight_checks(samples, extractor):
    """
    Validate batch uniqueness and freshness before launching workers.
    For very large result directories (>10k files), skip disk check to avoid I/O stalls.

    Parameters:
    -----------
    samples (list): List of simulation configs.
    extractor: Extractor-like object with optional sim_key and results_dir.

    Raises:
    -------
    RuntimeError: If already-evaluated samples are detected (in small directories only).
    """
    sim_keys = []
    for config in samples:
        try:
            key = extractor.sim_key_for_params(config)
        except Exception:
            key = None
        sim_keys.append(key)

    unique_keys = set(k for k in sim_keys if k is not None)
    if len(unique_keys) != len(sim_keys):
        print("\n[!] WARNING: Duplicate parametrizations detected in batch. This may waste compute.")

    existing = set()
    if hasattr(extractor, 'results_dir') and extractor.results_dir:
        # Fast heuristic: only enumerate directory if it's small enough to avoid I/O stalls
        # For massive datasets (>10k files), skip the check and rely on sim_id deduplication
        try:
            # Try a non-blocking stat to estimate directory size (ext4/xfs usually cache this)
            dir_stat = os.stat(extractor.results_dir)
            # Note: st_nlink is a weak proxy for file count, but avoids full enumeration
            estimated_files = max(2, dir_stat.st_nlink - 2)  # -2 for '.' and '..'
            
            if estimated_files > PREFLIGHT_CHECK_MAX_FILES:
                print(f"\n[i] Skipping preflight disk check (directory has ~{estimated_files} estimated files). "
                      f"Relying on sim_id deduplication in DataCollector.")
                return
            
            # Safe to enumerate: directory is still reasonably small
            all_files = set(os.listdir(extractor.results_dir))
        except FileNotFoundError:
            all_files = set()

        for key in unique_keys:
            if key is None:
                continue
            if f"{key}.json" in all_files:
                existing.add(key)
    
    if existing:
        raise RuntimeError(
            f"Pre-flight check failed: {len(existing)} samples already have results in {extractor.results_dir}"
        )


def get_adaptive_worker_count(max_cores=None):
    """
    Calculate a worker count based on current machine load.

    Parameters:
    -----------
    max_cores (int | None): Optional CPU core limit.

    Returns:
    --------
    int: Suggested worker process count.
    """
    if max_cores is None:
        max_cores = mp.cpu_count()
    try:
        load1, _, _ = os.getloadavg()
    except AttributeError:
        load1 = 0
    return max(1, int(max_cores - load1) - MIN_FREE_CORES)

def run_parallel_simulations(samples, extractor, n_workers, adaptive=False):
    """
    Run simulations in parallel and yield progress updates.
    
    Uses process-global extractor to avoid serialization overhead on large batches.
    For 6.4M+ point datasets, adaptive resizing is disabled to prevent pool.join() stalls.
    
    Parameters:
    -----------
    samples (list): List of configuration dictionaries.
    extractor: Configured Extractor instance.
    n_workers (int): Worker count for fixed mode.
    adaptive (bool): If True, re-evaluates worker count between micro-batches.
        Disabled automatically for very large batches (>100k samples).
        
    Yields:
    -------
    tuple: (completed_count, total_count, elapsed_time, result_data)
    """

    _run_preflight_checks(samples, extractor)

    # Fix #1: Pass only (index, config) to workers; extractor is in process globals
    task_args = [(i, config) for i, config in enumerate(samples)]

    total = len(samples)
    completed = 0
    start_time = time.time()
    
    # Fix #3: Disable adaptive resizing for massive batches to avoid pool.join() stalls
    use_adaptive = adaptive and total <= 100000
    if adaptive and total > 100000:
        print(f"\n[i] Batch size ({total} samples) exceeds 100k. Disabling adaptive resizing "
              f"to avoid pool.join() stalls. Using fixed mode with {n_workers} workers.")
        use_adaptive = False
    
    if not use_adaptive:
        # fixed mode: single pool for the entire batch
        chunk_size = max(1, total // (n_workers * 4))
        try:
            # Fix #1: Use initializer to copy extractor once per worker
            with mp.Pool(processes=n_workers, initializer=_worker_init, initargs=(extractor,)) as pool:
                for result in pool.imap_unordered(worker_task, task_args, chunksize=chunk_size):
                    success, payload = result
                    final_data = _format_worker_result(success, payload)
                    completed += 1
                    elapsed = time.time() - start_time
                    yield (completed, total, elapsed, final_data)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
    else:
        # adaptive mode: use persistent pool and resize only on meaningful load shifts
        remaining = list(task_args)
        current_workers = max(1, get_adaptive_worker_count())
        # Fix #1: Use initializer to copy extractor once per worker
        pool = mp.Pool(processes=current_workers, initializer=_worker_init, initargs=(extractor,))
        
        try:
            while remaining:
                micro_size = min(MICRO_BATCH_INTERVAL, len(remaining))
                micro_batch = remaining[:micro_size]
                remaining = remaining[micro_size:]
                
                chunk_size = max(1, micro_size // (current_workers * 4))

                for result in pool.imap_unordered(worker_task, micro_batch, chunksize=chunk_size):
                    success, payload = result
                    final_data = _format_worker_result(success, payload)
                    completed += 1
                    elapsed = time.time() - start_time
                    yield (completed, total, elapsed, final_data)

                if remaining:
                    ideal_workers = max(1, get_adaptive_worker_count())
                    if abs(ideal_workers - current_workers) >= ADAPTIVE_RESIZE_HYSTERESIS:
                        print(
                            f"\n[INFO] Load shifted. Resizing pool: "
                            f"{current_workers} -> {ideal_workers} workers.",
                            flush=True,
                        )
                        pool.close()
                        pool.join()
                        current_workers = ideal_workers
                        pool = mp.Pool(processes=current_workers, initializer=_worker_init, initargs=(extractor,))
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise KeyboardInterrupt
        finally:
            try:
                pool.close()
                pool.join()
            except Exception:
                pass
