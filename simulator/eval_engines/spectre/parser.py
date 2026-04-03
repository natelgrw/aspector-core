"""
parser.py

Author: natelgrw
Last Edited: 04/01/2026

Spectre parser for extracting PSF and CSV data into Python dictionaries.
"""

import os
import subprocess
import tempfile
import re
import fnmatch
import random
import time
import libpsf

# ===== Constants ===== #


IGNORE_LIST = ['*.primitives', '*.subckts', 'logFile']

OCEAN_TEMPLATE_BOTH = """
resultsDir = \"%(dir_file_path)s\"
openResults(resultsDir)
selectResult('tran)
wave_1 = v("Voutp")
wave_2 = v("Voutn")
ocnPrint(?output \"%(csv_output_path_voutp)s\" wave_1 ?precision 15)
ocnPrint(?output \"%(csv_output_path_voutn)s\" wave_2 ?precision 15)
exit"""

OCEAN_TEMPLATE_SINGLE = """
resultsDir = \"%(dir_file_path)s\"
openResults(resultsDir)
selectResult('tran)
wave_1 = v("Voutp")
ocnPrint(?output \"%(csv_output_path_voutp)s\" wave_1 ?precision 15)
exit"""


# ===== Custom Exceptions ===== #


class FileNotCompatible(Exception):
    """
    Exception class raised when a file is not compatible with libpsf parser.
    
    Initialization Parameters:
    --------------------------
    message (str): Description of the compatibility issue.
    """
    def __init__(self, *args, **kwargs):

        Exception.__init__(self,  args, kwargs)


# ===== Utility Functions ===== #


def is_ignored(string):
    """
    Checks if a filename matches any pattern in the ignore list.

    Parameters:
    -----------
    string (str): Filename to check against ignore patterns.

    Returns:
    --------
    bool: True if the filename matches any ignore pattern, False otherwise.
    """
    return any([fnmatch.fnmatch(string, pattern) for pattern in IGNORE_LIST])


def ocean_export_csv(dir_file_path, csv_output_path_voutp, csv_output_path_voutn, include_voutn=True):
    """
    Export transient signals to CSV using OCEAN script.
    
    Extracts voltage waveforms from PSF results and converts to CSV format.
    
    Parameters:
    -----------
    dir_file_path (str): Path to directory containing PSF results.
    csv_output_path_voutp (str): Output CSV path for positive voltage.
    csv_output_path_voutn (str): Output CSV path for negative voltage.
    include_voutn (bool): If True, export both Voutp and Voutn; else only Voutp.
    """
    template = OCEAN_TEMPLATE_BOTH if include_voutn else OCEAN_TEMPLATE_SINGLE
    
    # create temporary OCEAN script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ocn', delete=False) as tmp_script:
        script_path = tmp_script.name
        tmp_script.write(template % {
            "dir_file_path": dir_file_path,
            "csv_output_path_voutp": csv_output_path_voutp,
            "csv_output_path_voutn": csv_output_path_voutn
        })

    try:
        # retry with jitter to avoid synchronized license-server bursts
        max_retries = 4
        retry_jitter_s = (2.0, 5.0)
        last_err = "unknown error"

        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    ["ocean", "-nograph", "-restore", script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=100
                )
                if result.returncode == 0:
                    return
                last_err = result.stderr.strip() or f"non-zero return code: {result.returncode}"
            except subprocess.TimeoutExpired:
                last_err = "timeout after 100s"

            if attempt < (max_retries - 1):
                time.sleep(random.uniform(*retry_jitter_s))

        raise RuntimeError(f"OCEAN error after {max_retries} attempts: {last_err}")
    finally:
        os.remove(script_path)


def parse_ocean_csv(file_path, key_name, data_dict):
    """
    Parse OCEAN CSV export and append data to dictionary.
    
    Converts time to nanoseconds and voltage to millivolts.
    Modifies data_dict in place.
    """
    # time unit conversion: normalize to nanoseconds
    unit_to_multiplier_time = {
        "s": 1e9,      # seconds → nanoseconds
        "n": 1,        # nanoseconds → nanoseconds
        "p": 1e-3,     # picoseconds → nanoseconds
        "f": 1e-6,     # femtoseconds → nanoseconds
    }

    # voltage unit conversion: normalize to millivolts
    unit_to_multiplier_voltage = {
        "V": 1000,     # volts → millivolts
        "m": 1,        # millivolts → millivolts
        "u": 1e-3,     # microvolts → millivolts
    }

    # parse csv file line by line
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("time"):
                continue

            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue

            # parse time value and unit
            match_time = re.match(r"([\d\.\-Ee+]+)([a-z]*)", parts[0])
            if match_time:
                time_val = float(match_time.group(1))
                time_unit = match_time.group(2)
                time_ns = round(time_val * unit_to_multiplier_time.get(time_unit, 1), 8)

            # parse voltage value and unit
            match_volt = re.match(r"([\d\.\-Ee+]+)([a-z]*)", parts[1])
            if match_volt:
                volt_val = float(match_volt.group(1))
                volt_unit = match_volt.group(2)
                volt_mV = round(volt_val * unit_to_multiplier_voltage.get(volt_unit, 1), 8)

            data_dict.setdefault(key_name, []).append((time_ns, volt_mV))


# ===== Spectre Parser Class ===== #


class SpectreParser:
    """
    Parser for Spectre simulation results.
    Handles PSF (Post-Simulation Format) files and OCEAN CSV exports.
    """

    @classmethod
    def parse(cls, raw_folder):
        """
        Parse all simulation results from a results folder.
        
        Extracts phase margin, ugbw from margin.stb files if present.
        """
        folder_path = os.path.abspath(raw_folder)
        data = dict()
        try:
            files = os.listdir(folder_path)
        except FileNotFoundError:
            return data

        for file in files:
            if is_ignored(file):
                continue

            file_path = os.path.join(raw_folder, file)

            # handle transient result files
            if file.endswith(".tran.tran"):
                base_name = file.replace(".tran.tran", "")
                output_csv_voutp = os.path.join(folder_path, f"{base_name}_Voutp.csv")
                output_csv_voutn = os.path.join(folder_path, f"{base_name}_Voutn.csv")
                if not os.path.exists(output_csv_voutp):
                    try:
                        try:
                            ocean_export_csv(file_path, output_csv_voutp, output_csv_voutn, include_voutn=True)
                            has_voutn = True
                        except RuntimeError:
                            ocean_export_csv(file_path, output_csv_voutp, output_csv_voutn, include_voutn=False)
                            has_voutn = False
                    except Exception:
                        continue
                else:
                    has_voutn = os.path.exists(output_csv_voutn)
                tran_data = []
                parse_ocean_csv(output_csv_voutp, "temp_key", { "temp_key": tran_data })
                tran_dict = {}
                if tran_data:
                    times, vouts = zip(*tran_data)
                    tran_dict['time'] = list(times)
                    tran_dict['Voutp'] = list(vouts)
                else:
                    tran_dict['time'] = []
                    tran_dict['Voutp'] = []
                if has_voutn and os.path.exists(output_csv_voutn):
                    tran_data_n = []
                    parse_ocean_csv(output_csv_voutn, "temp_key", { "temp_key": tran_data_n })
                    if tran_data_n:
                        _, vouts_n = zip(*tran_data_n)
                        tran_dict['Voutn'] = list(vouts_n)
                data[base_name] = tran_dict
                continue

            # process other psf files
            try:
                datum = cls.process_file(file_path)
            except (FileNotCompatible, RuntimeError, Exception):
                continue
            _, kwrd = os.path.split(file)
            if not kwrd.endswith('.fd'):
                kwrd = os.path.splitext(kwrd)[0]
            data[kwrd] = datum
        return data

    @classmethod
    def process_file(cls, file):
        """
        Process a single PSF result file.
        
        Extracts signal data using libpsf; includes sweep parameters if present.
        
        Parameters:
        -----------
        file (str): Path to the PSF file.
        
        Returns:
        --------
        datum (dict): Signal data and sweep information.
        """
        fpath = os.path.abspath(file)
        try:
            psfobj = libpsf.PSFDataSet(fpath)
        except:
            raise FileNotCompatible('file {} was not compatible with libpsf'.format(file))

        is_swept = psfobj.is_swept()
        datum = dict()
        
        # extract all signal data; skip signals that cause libpsf errors
        for signal in psfobj.get_signal_names():
            try:
                datum[signal] = psfobj.get_signal(signal)
            except (RuntimeError, Exception):
                continue

        # include sweep parameters if present
        if is_swept:
            try:
                datum['sweep_vars'] = psfobj.get_sweep_param_names()
                datum['sweep_values'] = psfobj.get_sweep_values()
            except (RuntimeError, Exception):
                pass

        psfobj.close()
        return datum
