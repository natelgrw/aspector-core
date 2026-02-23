"""
collector.py

Data Collection module for Mass Data Generation.
Buffers simulation results into intermediate JSON files and merges them into a final Parquet file.
"""

import os
import json
import uuid
import pandas as pd
import time
import glob

class DataCollector:
    def __init__(self, output_dir, buffer_size=1000, parquet_name="dataset.parquet"):
        """
        Initialize DataCollector.
        
        Args:
            output_dir (str): Directory to save intermediate files and final parquet.
            buffer_size (int): Number of records to hold in memory before flush.
            parquet_name (str): Name of the final parquet file.
        """
        self.output_dir = output_dir
        self.buffer_size = buffer_size
        self.buffer = []
        self.parquet_name = parquet_name
        self.dataset_path = os.path.join(output_dir, parquet_name)
        
        os.makedirs(output_dir, exist_ok=True)
        
    def log(self, config, specs, meta=None, operating_points=None):
        """
        Log a single simulation result.
        
        Args:
            config (dict): Input parameters (Sizing + Env).
            specs (dict): Output specifications.
            meta (dict, optional): Metadata (sim_id, timestamp).
            operating_points (dict, optional): Operating points data.
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
                # Handle tuples (like swing [min, max]) by converting to scalar or string
                if isinstance(v, (list, tuple)):
                    record[f"out_{k}_min"] = v[0]
                    record[f"out_{k}_max"] = v[1]
                    record[f"out_{k}_val"] = abs(v[1] - v[0])
                else:
                    record[f"out_{k}"] = v
        else:
            record['valid'] = False
            
        # Add Operating Points
        if operating_points:
            for comp, params in operating_points.items():
                for param_name, val in params.items():
                    record[f"op_{comp}_{param_name}"] = val
                    
        if meta:
            # Remove 'env' if present to avoid redundancy/errors
            meta_copy = dict(meta)
            meta_copy.pop('env', None)
            record.update(meta_copy)
            
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
        Write buffer to an intermediate JSON file.
        """
        if not self.buffer:
            return
            
        batch_id = str(uuid.uuid4())
        batch_filename = f"batch_{batch_id}.json"
        batch_path = os.path.join(self.output_dir, batch_filename)
        
        try:
            with open(batch_path, 'w') as f:
                json.dump(self.buffer, f, indent=2)
        except Exception as e:
            print(f"[!] Failed to write batch {batch_filename}: {e}")
            
        self.buffer = [] # Clear buffer

    def finalize(self):
        """
        Merges all intermediate JSON files into a single Parquet file and cleans up.
        """
        # Flush any remaining data
        self.flush()
        
        json_pattern = os.path.join(self.output_dir, "batch_*.json")
        json_files = glob.glob(json_pattern)
        
        if not json_files:
            print("No data collected to merge.")
            return

        all_data = []
        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                    all_data.extend(data)
            except Exception as e:
                print(f"[!] Error reading {jf}: {e}")
        
        if all_data:
            df = pd.DataFrame(all_data)
            
            # Ensure topology_id/netlist_name is present for grouping
            if 'netlist_name' not in df.columns and 'topology_id' in df.columns:
                df['netlist_name'] = df['topology_id']
                
            # Group by topology and save separate parquet files
            if 'netlist_name' in df.columns:
                topologies = df['netlist_name'].unique()
                for topo in topologies:
                    # Filter data for this topology
                    topo_df = df[df['netlist_name'] == topo].copy()
                    
                    # Drop columns that are entirely NaN (parameters not in this topology)
                    topo_df = topo_df.dropna(axis=1, how='all')
                    
                    # Create topology-specific parquet name
                    topo_parquet_name = f"{topo}_{self.parquet_name}"
                    topo_dataset_path = os.path.join(self.output_dir, topo_parquet_name)
                    
                    try:
                        if os.path.exists(topo_dataset_path):
                             existing_df = pd.read_parquet(topo_dataset_path)
                             topo_df = pd.concat([existing_df, topo_df], ignore_index=True)

                        topo_df.to_parquet(topo_dataset_path)
                        print(f"Successfully wrote {len(topo_df)} records to {topo_dataset_path}")
                    except Exception as e:
                        print(f"[!] Error writing Parquet for {topo}: {e}")
            else:
                # Fallback if no topology info (shouldn't happen)
                try:
                    if os.path.exists(self.dataset_path):
                         existing_df = pd.read_parquet(self.dataset_path)
                         df = pd.concat([existing_df, df], ignore_index=True)

                    df.to_parquet(self.dataset_path)
                    print(f"Successfully wrote {len(df)} records to {self.dataset_path}")
                except Exception as e:
                    print(f"[!] Error writing Parquet: {e}")
                
            # Cleanup
            for jf in json_files:
                os.remove(jf)
            print("Cleaned up intermediate files.")
