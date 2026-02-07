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
        extractor(config, sim_id=idx)
        return True, None
    except Exception as e:
        # Return error info instead of writing to file directly if possible,
        # or write to discrete log.
        err_msg = f"Sim {idx} Failed: {str(e)}"
        return False, err_msg

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
        (completed_count, total_count, elapsed_time)
    """
    
    task_args = []
    for i, config in enumerate(samples):
        task_args.append((i + 1, config, extractor))

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
                success, error = result
                
                if not success and error:
                    # Ideally log this error to a file
                    with open("simulation_errors.log", "a") as f:
                        f.write(error + "\n")
                
                completed += 1
                elapsed = time.time() - start_time
                
                yield (completed, total, elapsed)
                
    except KeyboardInterrupt:
        # Re-raise to let caller handle cleanup/exit
        raise KeyboardInterrupt
