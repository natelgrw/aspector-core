"""
collector.py

Author: natelgrw
Last Edited: 04/01/2025

Data Collection module for Mass Data Generation.
Buffers simulation results into intermediate JSON files and merges them into a final Parquet file.
"""

import os
import json
import uuid
import math
import pandas as pd
import time
import glob
import re
from collections import defaultdict

OP_KEY_RE = re.compile(r'^op_(MM\d+)_(.+)$')

# Immediate flush threshold to prevent OOM with large datasets (6.4M+ points)
IMMEDIATE_FLUSH_THRESHOLD = 5000

CANONICAL_SPEC_KEYS = [
    'estimated_area_um2',
    'cmrr_dc_db',
    'gain_ol_dc_db',
    'integrated_noise_vrms',
    'output_voltage_swing_range_v',
    'output_voltage_swing_min_v',
    'output_voltage_swing_max_v',
    'pm_deg',
    'power_w',
    'v_cm_ctrl',
    'psrr_dc_db',
    'settle_time_small_ns',
    'settle_time_large_ns',
    'slew_rate_v_us',
    'thd_db',
    'ugbw_hz',
    'vos_v',
]

CANONICAL_OP_PARAMS = [
    'region_of_operation', 'ids', 'vds', 'vgs', 'gm', 'gds', 'vth', 'vdsat',
    'cgg', 'cgs', 'cdd', 'cgd', 'css',
    'noise_therm_rs_vrms', 'noise_therm_rd_vrms', 'noise_therm_rg_vrms',
    'noise_therm_sid_vrms', 'noise_flicker_vrms',
    'noise_shot_igb_vrms', 'noise_shot_igd_vrms', 'noise_shot_igs_vrms',
    'fin_strength_risk', 'pelgrom_coefficient', 'stress_sensitivity', 'bandwidth_jitter_risk'
]

class DataCollector:

    @staticmethod
    def _to_json_safe(value):
        """
        Recursively convert NaN/Inf values to None so JSON stays standards-compliant.
        """
        if isinstance(value, dict):
            return {k: DataCollector._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [DataCollector._to_json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [DataCollector._to_json_safe(v) for v in value]
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
        return value

    def __init__(self, output_dir, buffer_size=1000, parquet_name="dataset.parquet", max_rows_per_file=50000):
        """
        Initialize the DataCollector class.
        
        Parameters:
        -----------
        output_dir (str): Directory to save intermediate files and final parquet.
        buffer_size (int): Number of records to hold in memory before flush.
        parquet_name (str): Name of the final parquet file.
        max_rows_per_file (int): Maximum number of rows per output parquet file.
        """
        self.output_dir = output_dir
        self.buffer_size = buffer_size
        self.buffer = []
        self.parquet_name = parquet_name
        self.dataset_path = os.path.join(output_dir, parquet_name)
        self.max_rows_per_file = max_rows_per_file
        self.topology_op_components = defaultdict(set)
        
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def _extract_mm_components_from_record(record):
        """
        Extract transistor component names (MM0, MM1, ...) from flattened op keys.
        """
        mm_components = set()
        if not isinstance(record, dict):
            return mm_components
        for key in record.keys():
            match = OP_KEY_RE.match(str(key))
            if match:
                mm_components.add(match.group(1))
        return mm_components

    def _update_topology_components(self, topo, record=None, operating_points=None):
        """
        Track all MM component names observed for a topology.
        """
        if topo is None:
            topo = 'unknown'

        if operating_points:
            for comp in operating_points.keys():
                if isinstance(comp, str) and comp.startswith('MM'):
                    self.topology_op_components[topo].add(comp)

        if record:
            self.topology_op_components[topo].update(self._extract_mm_components_from_record(record))

    def _pad_record_schema(self, record, topo):
        """
        Ensure a record conforms to the topology-wide spec and operating-point schema.
        """
        nan_value = float('nan')
        mm_components = set(self.topology_op_components.get(topo, set()))
        mm_components.update(self._extract_mm_components_from_record(record))

        for spec_key in CANONICAL_SPEC_KEYS:
            record.setdefault(f'out_{spec_key}', nan_value)
        record.setdefault('out_output_voltage_swing_range_v_val', nan_value)

        for mm_name in mm_components:
            for param_name in CANONICAL_OP_PARAMS:
                record.setdefault(f'op_{mm_name}_{param_name}', nan_value)

        return record
        
    def log(self, config, specs, meta=None, operating_points=None):
        """
        Log a single simulation result.
        
        Parameters:
        -----------
        config (dict): Input parameters (Sizing + Env).
        specs (dict): Output specifications.
        meta (dict, optional): Metadata (sim_id, timestamp).
        operating_points (dict, optional): Operating points data.
        """
        record = {}
        nan_value = float('nan')
        
        # add input configuration
        for k, v in config.items():
            record[f"in_{k}"] = v

        # Determine is_diff: explicit meta flag takes precedence over netlist name parsing
        if meta and 'is_diff' in meta:
            is_diff = bool(meta['is_diff'])
        else:
            # Fallback to netlist name parsing (less reliable for large-scale data)
            try:
                netname = meta.get('netlist_name') if meta else None
                netname = netname or record.get('netlist_name') or ''
                is_diff = True if isinstance(netname, str) and 'differential' in netname.lower() else False
            except Exception:
                is_diff = False
        
        record['is_diff'] = is_diff

        if is_diff:
            required_inputs = {
                'fet_num': 0,
                'vdd': 0.0,
                'vcm': 0.0,
                'tempc': 25,
                'cload_val': 0.0,
                'vbiasp0': 0.0,
                'vbiasn0': 0.0,
                'nA1': 0.0,
                'nA2': 0.0,
                'nA3': 0.0,
                'nB1': 0,
                'nB2': 0,
                'nB3': 0,
                'run_gatekeeper': 0,
                'run_full_char': 0
            }
            for k, default in required_inputs.items():
                in_key = f'in_{k}'
                if in_key not in record:
                    record[in_key] = default
            
        # add output specs
        sim_status = meta.get('sim_status') if meta else None
        is_fully_valid = (sim_status == 2)

        if specs:
            record['valid'] = is_fully_valid
            for k, v in specs.items():
                if k == 'valid': continue
                if isinstance(v, (list, tuple)):
                    record[f"out_{k}_min"] = v[0]
                    record[f"out_{k}_max"] = v[1]
                    record[f"out_{k}_val"] = abs(v[1] - v[0])
                else:
                    record[f"out_{k}"] = v
        else:
            record['valid'] = False
            
        # add operating points
        if operating_points:
            for comp, params in operating_points.items():
                for param_name, val in params.items():
                    record[f"op_{comp}_{param_name}"] = val
                    
        if meta:
            meta_copy = dict(meta)
            meta_copy.pop('env', None)
            record.update(meta_copy)
            
        record['timestamp'] = time.time()
        # always add algorithm and netlist_name if present in meta
        if meta:
            if 'algorithm' in meta:
                record['algorithm'] = meta['algorithm']
            if 'netlist_name' in meta:
                record['netlist_name'] = meta['netlist_name']

        topo_name = record.get('netlist_name') or 'unknown'
        self._update_topology_components(topo_name, record=record, operating_points=operating_points)

        try:
            is_diff = record.get('is_diff', False)
        except Exception:
            is_diff = False

        if not record['valid']:
            for key in list(record.keys()):
                if key.startswith('out_') and not key.endswith('_valid'):
                    record[key] = nan_value
                if key.startswith('op_'):
                    record[key] = nan_value

            for spec_key in CANONICAL_SPEC_KEYS:
                out_key = f'out_{spec_key}'
                if out_key not in record:
                    record[out_key] = nan_value
            if 'out_output_voltage_swing_range_v_val' not in record:
                record['out_output_voltage_swing_range_v_val'] = nan_value

            if is_diff:
                for mm in sorted(self.topology_op_components.get(topo_name, set())):
                    for param_name in CANONICAL_OP_PARAMS:
                        key = f'op_{mm}_{param_name}'
                        if key not in record:
                            record[key] = nan_value

        self._pad_record_schema(record, topo_name)

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
            safe_buffer = [self._to_json_safe(rec) for rec in self.buffer]
            with open(batch_path, 'w') as f:
                json.dump(safe_buffer, f, indent=2, allow_nan=False)
        except Exception as e:
            print(f"[!] Failed to write batch {batch_filename}: {e}")
            
        self.buffer = [] 

    def finalize(self, discard_partial=True, preserve_json=True):
        """
        Merges all intermediate JSON files into a single Parquet file and cleans up.
        
        Parameters:
        -----------
        discard_partial (bool): If True, discard any unflushed buffer (< buffer_size). This prevents duplicates on 
                                resume since the Sobol/TuRBO state checkpoint only advances on full-batch boundaries.
        preserve_json (bool): If True, keep intermediate JSON files for debugging. Otherwise, remove them after aggregation.
        """
        # handle partial buffer
        if discard_partial:
            if self.buffer:
                print(f"[i] Discarding {len(self.buffer)} buffered records (partial batch).")
            self.buffer = []
        else:
            self.flush()

        # find all JSON and JSONL files under output_dir
        json_files = []
        json_files.extend(sorted(glob.glob(os.path.join(self.output_dir, "**", "*.json"), recursive=True)))
        json_files.extend(sorted(glob.glob(os.path.join(self.output_dir, "**", "*.jsonl"), recursive=True)))
        # remove duplicates and sort
        json_files = sorted(list(dict.fromkeys(json_files)))

        if not json_files:
            print("No JSON files found to aggregate into Parquet.")
            return

        # streaming aggregation
        buffers = defaultdict(list) 
        seen_ids = defaultdict(set)
        jf_remaining = {} 
        part_idx = defaultdict(lambda: 1) 

        # inspect existing parquet parts in output_dir and initialize part indices
        try:
            parquet_files = sorted(glob.glob(os.path.join(self.output_dir, "*.parquet")))
            part_re = re.compile(r"(?P<topo>.+)_(?P<idx>\d+)\.parquet$")
            for p in parquet_files:
                bn = os.path.basename(p)
                m = part_re.match(bn)
                if not m:
                    continue
                topo = m.group('topo')
                try:
                    idx = int(m.group('idx'))
                    if idx >= part_idx[topo]:
                        part_idx[topo] = idx + 1
                except Exception:
                    pass
                try:
                    existing_ids = pd.read_parquet(p, columns=['sim_id'])
                    if 'sim_id' in existing_ids.columns:
                        for sid in existing_ids['sim_id'].dropna().unique():
                            seen_ids[topo].add(sid)
                except Exception:
                    pass
        except Exception:
            pass

        # initialize seen_ids from existing parquet parts and pre-scan topology schemas
        for jf in json_files:
            if os.path.sep + 'markers' + os.path.sep in jf:
                continue
            try:
                if jf.lower().endswith('.jsonl'):
                    with open(jf, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            jf_remaining[jf] = jf_remaining.get(jf, 0) + 1
                            try:
                                item = json.loads(line)
                            except Exception:
                                continue
                            topo = item.get('netlist_name') or item.get('requested_name') or 'unknown'
                            self._update_topology_components(topo, record=item)
                else:
                    with open(jf, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            jf_remaining[jf] = len(data)
                            for item in data:
                                if isinstance(item, dict):
                                    topo = item.get('netlist_name') or item.get('requested_name') or 'unknown'
                                    self._update_topology_components(topo, record=item)
                        else:
                            jf_remaining[jf] = 1
                            if isinstance(data, dict):
                                topo = data.get('netlist_name') or data.get('requested_name') or 'unknown'
                                self._update_topology_components(topo, record=data)
            except Exception:
                jf_remaining[jf] = jf_remaining.get(jf, 0)

        def _write_part(topo, items):
            """
            Write a part file for a given topology with the provided items.
            Also handles cleanup of source JSON files if all their records are consumed.

            Parameters:
            -----------
            topo (str): Topology name for this part.
            items (list): List of (record, source_json_file) tuples to write in this part.

            Returns:
            --------
            (bool, str or None): (success flag, error message if failed)
            """
            nonlocal part_idx
            if not items:
                return True, None
            records = [self._pad_record_schema(rec, topo) for (rec, _jf) in items]
            try:
                df_part = pd.DataFrame(records)
                # Enforce numeric dtypes for measurement columns so missing values
                # are represented consistently as NaN when read back with pandas.
                numeric_prefixes = ('out_', 'op_')
                for col in df_part.columns:
                    if col.startswith(numeric_prefixes):
                        df_part[col] = pd.to_numeric(df_part[col], errors='coerce')
            except Exception as e:
                return False, f"build_df_failed: {e}"

            # sanitize topology name
            safe_topo = str(topo).replace(' ', '_').replace('/', '_')
            idx = part_idx[topo]
            out_name = f"{safe_topo}_{idx}.parquet"
            out_path = os.path.join(self.output_dir, out_name)
            tmp_path = out_path + ".tmp"
            try:
                df_part.to_parquet(tmp_path)
                os.replace(tmp_path, out_path)
                print(f"Successfully wrote {len(df_part)} records to {out_path}")
                part_idx[topo] += 1
                # decrement jf_remaining for source files and remove files fully consumed
                for (_rec, src) in items:
                    if src in jf_remaining:
                        jf_remaining[src] = max(0, jf_remaining[src] - 1)
                        if jf_remaining[src] == 0 and not preserve_json:
                            try:
                                os.remove(src)
                            except Exception:
                                pass
                return True, None
            except Exception as e:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                return False, str(e)

        # stream through JSON files and route records into per-topo buffers
        # Note: Sim ID deduplication (seen_ids) is enabled for safety during resume scenarios.
        # For trusted sequential workflows (Sobol with atomic checkpoints), deduplication overhead
        # can be disabled by setting use_dedup=False (future enhancement).
        
        record_count_since_flush = 0
        
        for jf in json_files:
            if os.path.sep + 'markers' + os.path.sep in jf:
                continue
            try:
                if jf.lower().endswith('.jsonl'):
                    with open(jf, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                item = json.loads(line)
                            except Exception:
                                continue
                            if isinstance(item, dict) and set(item.keys()) <= {'id','sim_id','sim_status','netlist','timestamp'}:
                                continue
                            if isinstance(item, dict) and set(item.keys()) <= {'requested_name','state'}:
                                continue
                            topo = item.get('netlist_name') or item.get('requested_name') or 'unknown'
                            sid = item.get('sim_id') or item.get('id')
                            if sid is not None and sid in seen_ids[topo]:
                                # already seen
                                continue
                            if sid is not None:
                                seen_ids[topo].add(sid)
                            buffers[topo].append((item, jf))
                            record_count_since_flush += 1
                            
                            # Immediate flush to prevent OOM on large datasets (6.4M+ points)
                            if record_count_since_flush >= IMMEDIATE_FLUSH_THRESHOLD:
                                for topo_flush, items_flush in list(buffers.items()):
                                    if len(items_flush) > 0:
                                        ok, err = _write_part(topo_flush, items_flush)
                                        if not ok:
                                            print(f"[!] Failed to write part for {topo_flush}: {err}")
                                        else:
                                            buffers[topo_flush] = []
                                record_count_since_flush = 0
                else:
                    with open(jf, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    if set(item.keys()) <= {'id','sim_id','sim_status','netlist','timestamp'}:
                                        continue
                                    if set(item.keys()) <= {'requested_name','state'}:
                                        continue
                                topo = item.get('netlist_name') or item.get('requested_name') or 'unknown'
                                sid = item.get('sim_id') or item.get('id')
                                if sid is not None and sid in seen_ids[topo]:
                                    continue
                                if sid is not None:
                                    seen_ids[topo].add(sid)
                                buffers[topo].append((item, jf))
                                record_count_since_flush += 1
                                
                                # Immediate flush every ~5000 records to prevent OOM
                                if record_count_since_flush >= IMMEDIATE_FLUSH_THRESHOLD:
                                    for topo_flush, items_flush in list(buffers.items()):
                                        if len(items_flush) > 0:
                                            ok, err = _write_part(topo_flush, items_flush)
                                            if not ok:
                                                print(f"[!] Failed to write part for {topo_flush}: {err}")
                                            else:
                                                buffers[topo_flush] = []
                                    record_count_since_flush = 0
                        elif isinstance(data, dict):
                            item = data
                            if 'specs' not in item and 'netlist_name' not in item and set(item.keys()) <= {'id','sim_id','sim_status','netlist','timestamp'}:
                                continue
                            if 'requested_name' in item and 'state' in item and 'netlist_name' not in item:
                                continue
                            topo = item.get('netlist_name') or item.get('requested_name') or 'unknown'
                            sid = item.get('sim_id') or item.get('id')
                            if sid is not None and sid in seen_ids[topo]:
                                continue
                            if sid is not None:
                                seen_ids[topo].add(sid)
                            buffers[topo].append((item, jf))
                            record_count_since_flush += 1
                            
                            # Immediate flush for OOM prevention
                            if record_count_since_flush >= IMMEDIATE_FLUSH_THRESHOLD:
                                for topo_flush, items_flush in list(buffers.items()):
                                    if len(items_flush) > 0:
                                        ok, err = _write_part(topo_flush, items_flush)
                                        if not ok:
                                            print(f"[!] Failed to write part for {topo_flush}: {err}")
                                        else:
                                            buffers[topo_flush] = []
                                record_count_since_flush = 0
            except Exception as e:
                print(f"[!] Error reading {jf}: {e}")

        # after all JSON files processed, flush remaining buffers to final parts
        for topo, items in list(buffers.items()):
            if not items:
                continue
            # write in chunks of max_rows_per_file
            i = 0
            while i < len(items):
                chunk = items[i:i + self.max_rows_per_file]
                ok, err = _write_part(topo, chunk)
                if not ok:
                    print(f"[!] Failed to write final part for {topo}: {err}")
                    break
                i += self.max_rows_per_file

        if preserve_json:
            print("Preserving intermediate JSON files for debugging (preserve_json=True).")
            return

        # remove any remaining json files that weren't removed during part writes
        for jf, rem in list(jf_remaining.items()):
            if rem == 0:
                # already removed
                continue
            try:
                os.remove(jf)
            except Exception:
                pass
        print("Cleaned up intermediate JSON files.")
