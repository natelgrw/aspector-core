"""
runner.py

Handles parallel execution of circuit simulations.
Separates compute logic from CLI user interface.
"""

import multiprocessing as mp
import time
import os

def worker_task(args):
    """
    Worker function executed by multiprocessing Pool.
    Receives (index, config, tokenizer (extractor)).
    """
    idx, config, extractor = args
    try:
        # Run extraction
        val = extractor(config, sim_id=idx)
        return True, (idx, val)
    except Exception as e:
        # Return error info instead of writing to file directly if possible,
        # or write to discrete log.
        err_msg = f"Sim {idx} Failed: {str(e)}"
        return False, (idx, err_msg)

def run_parallel_simulations(samples, extractor, n_workers):
    """
    Orchestrates the parallel execution of the simulation batch.
    Yields progress updates to the caller.
    
    Parameters:
    -----------
    samples : list
        List of configuration dictionaries (Sobol samples).
    extractor : Extractor
        Configured Extractor instance.
    n_workers : int
        Number of parallel processes.
        
    Yields:
    -------
    tuple
        (completed_count, total_count, elapsed_time, result_data)
        result_data is (index, (reward, specs)) or None if failed
    """
    
    task_args = []
    for i, config in enumerate(samples):
        # i is 0-based index in samples list. 
        # sim_id will be i+1 for logging, but we return i for array indexing
        task_args.append((i, config, extractor))

    total = len(samples)
    completed = 0
    start_time = time.time()
    
    # We use a chunksize heuristic to keep workers busy but responsive
    chunk_size = max(1, len(samples) // (n_workers * 4))

    try:
        with mp.Pool(processes=n_workers) as pool:
            # We use imap_unordered to get results as they finish 
            # for real-time progress reporting
            for result in pool.imap_unordered(worker_task, task_args, chunksize=chunk_size):
                success, payload = result
                
                # Payload is now (idx, data) for success, or (idx, err) for fail
                idx, data = payload
                
                if not success:
                    # Ideally log this error to a file
                    with open("simulation_errors.log", "a") as f:
                        f.write(f"ID {idx}: {data}\n")
                    final_data = (idx, (0.0, {'valid': False})) # Default fail with ID
                else:
                    final_data = (idx, data)
                
                completed += 1
                elapsed = time.time() - start_time
                
                yield (completed, total, elapsed, final_data)
                
    except KeyboardInterrupt:
        # Re-raise to let caller handle cleanup/exit
        raise KeyboardInterrupt
