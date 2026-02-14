"""
collector.py

Data Collection module for Mass Data Generation.
Buffers simulation results and appends them to Parquet files for NeurIPS-standard datasets.
"""

import os
import pandas as pd
import time

class DataCollector:
    def __init__(self, output_dir, buffer_size=100, parquet_name="dataset.parquet"):
        """
        Initialize DataCollector.
        
        Args:
            output_dir (str): Directory to save parquet files.
            buffer_size (int): Number of records to hold in memory before flush.
            parquet_name (str): Name of the parquet file to write to.
        """
        self.output_dir = output_dir
        self.buffer_size = buffer_size
        self.buffer = []
        self.dataset_path = os.path.join(output_dir, parquet_name)
        
        os.makedirs(output_dir, exist_ok=True)
        
    def log(self, config, specs, meta=None):
        """
        Log a single simulation result.
        
        Args:
            config (dict): Input parameters (Sizing + Env).
            specs (dict): Output specifications.
            meta (dict, optional): Metadata (sim_id, timestamp).
        """
        record = {}
        
        # Add Input Config
        for k, v in config.items():
            record[f"in_{k}"] = v
            
        # Add Output Specs
        if specs:
            valid = specs.get('valid', False)
            record['valid'] = valid
            for k, v in specs.items():
                if k == 'valid': continue
                # Handle tuples (like swing [min, max]) by converting to scalar or string?
                # For Parquet, better to scalarize or use specific columns
                if isinstance(v, (list, tuple)):
                    record[f"out_{k}_min"] = v[0]
                    record[f"out_{k}_max"] = v[1]
                    record[f"out_{k}_val"] = abs(v[1] - v[0])
                else:
                    record[f"out_{k}"] = v
        else:
            record['valid'] = False
            
        if meta:
            # Remove 'env' if present
            meta = dict(meta)
            meta.pop('env', None)
            record.update(meta)
        record['timestamp'] = time.time()
        # Always add algorithm and netlist_name if present in meta
        if meta:
            if 'algorithm' in meta:
                record['algorithm'] = meta['algorithm']
            if 'netlist_name' in meta:
                record['netlist_name'] = meta['netlist_name']
        
        self.buffer.append(record)
        
        if len(self.buffer) >= self.buffer_size:
            self.flush()
            
    def flush(self):
        """
        Write buffer to Parquet file.
        """
        if not self.buffer:
            return
            
        new_df = pd.DataFrame(self.buffer)
        
        # Append logic
        if os.path.exists(self.dataset_path):
            try:
                # Use fastparquet for append if available, else read-concat-write
                # Simplest robust way for now: Read, Concat, Write
                # Note: For massive datasets, this is slow. 
                # Ideally use per-batch parquet files (dataset_part_X.parquet) 
                # but user asked for "append to a .parquet file".
                # Partitioning is better. Let's do partitioning to avoid O(N^2) IO.
                
                # Timestamp-based partition file
                part_name = f"part_{int(time.time()*1000)}.parquet"
                part_path = os.path.join(self.output_dir, part_name)
                new_df.to_parquet(part_path)
                
                # Also try to maintain a master summary? No, huge files are bad.
                # User asked for "append to a very well formatted .parquet file".
                # Single file append is tricky. 
                # Let's try to overwrite the master file by concatenation for now 
                # if size is small, or use fastparquet append=True.
                
                # Try generic append using fastparquet (engine='fastparquet')
                # If fail, use separate file.
                try: 
                    new_df.to_parquet(self.dataset_path, engine='fastparquet', append=True)
                except:
                    # Fallback: Read-Concat
                    existing = pd.read_parquet(self.dataset_path)
                    combined = pd.concat([existing, new_df], ignore_index=True)
                    combined.to_parquet(self.dataset_path)

            except Exception as e:
                print(f"[!] Flush failed: {e}")
        else:
            new_df.to_parquet(self.dataset_path)
            
        self.buffer = [] # Clear buffer
