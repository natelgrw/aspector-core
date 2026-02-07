"""
differential_meas_man.py

Author: natelgrw
Last Edited: 02/06/2026

Measurement manager for processing and calculating performance specs 
for differential op-amp simulations.
"""

from simulator.eval_engines.spectre.core import EvaluationEngine
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt
import scipy.integrate as scint
from simulator import globalsy
from simulator.eval_engines.spectre.measurements.spec_functions import SpecCalc

# ===== Differential Op-Amp Measurement Manager ===== #

class OpampMeasMan(EvaluationEngine):
    """
    Measurement manager for differential op-amp simulations.
    Supports the calculation of performance specs including:
    - Gain
    - UGBW
    - Phase Margin
    - Power Consumption
    - CMRR
    - PSRR
    - Input Offset Voltage (Vos)
    - Linearity Range
    - Output Voltage Swing
    - Integrated Noise
    - Slew Rate
    - Settling Time
    - THD
    """
    
    # Inherit capability to calculate specs directly
    def process_ac(self, results, params):
        return ACTB.process_ac(results, params)

    def __init__(self, yaml_fname):
        EvaluationEngine.__init__(self, yaml_fname)

    def get_specs(self, results_dict, params):
        """
        Constructs a cleaned specs dictionary from an input results dictionary.
        """
        # print(f"DEBUG: get_specs params keys: {list(params.keys()) if params else 'None'}")
        
        # Flatten results if wrapped in netlist name dict (core.py behavior)
        if results_dict and isinstance(results_dict, dict):
             keys = list(results_dict.keys())
             # Heuristic: if key is a netlist name and value is tuple (state, specs, info)
             if len(keys) == 1 and isinstance(results_dict[keys[0]], tuple):
                  # Extract the actual specs dict from the tuple
                  results_dict = results_dict[keys[0]][1]

        # Check if results are already processed (Idempotency)
        if 'gain_ol' in results_dict or 'ugbw' in results_dict:
             return results_dict

        specs_dict = dict()
        # In differential scenarios, results might come differently structured 
        # depending on wrapper. Assuming standard tuple return from core.py
        if 'ac_dc' in results_dict:
             ac_dc_tuple = results_dict['ac_dc']
             specs_dict = ac_dc_tuple[1]
        else:
             # If direct dictionary passed
             return self.process_ac(results_dict, params)
             
        return specs_dict

    def compute_penalty(self, spec_nums, spec_kwrd):
        """
        Computes penalties for given spec numbers based on predefined spec ranges.
        """
        if type(spec_nums) is not list:
            spec_nums = [spec_nums]
        penalties = []
        for spec_num in spec_nums:
            penalty = 0
            if spec_kwrd in self.spec_range:
                spec_min, spec_max, w = self.spec_range[spec_kwrd]
                if spec_max is not None:
                    if spec_num > spec_max:
                        penalty += w * abs(spec_num - spec_max) / abs(spec_num)
                if spec_min is not None:
                    if spec_num < spec_min:
                        penalty += w * abs(spec_num - spec_min) / abs(spec_min)
            penalties.append(penalty)
        return penalties

# ===== AC Analysis Function Class for OpampMeasMan ===== #

class ACTB(object):
    """
    AC Analysis Trait Base for OpampMeasMan.
    """

    @classmethod
    def process_ac(self, results, params):
        """
        Processes AC analysis results to compute performance specifications.
        """
        # --- Safely Extract Results ---
        # Debugging hook for result type issues
        if not isinstance(results, dict):
             # print(f"DEBUG: results is type {type(results)}. Expected dict.")
             # If it's a list, it might be due to parser error or mismatch
             return {}

        # AC Analysis (Single Ended)
        # Prioritize STB loop for stability/gain analysis.
        # DO NOT include xf_sim here, as it contains transfer functions, not loop gain.
        # Open Loop Analysis (prioritize stb_ol)
        ac_result_se = results.get('stb_ol') or results.get('acswp-000_ac') or results.get('stb_loop') or results.get('ac')
        # ac_result_cm = results.get('acswp-001_ac')
        dc_results = results.get('dcOp_sim') or results.get('dcswp-500_dcOp') or results.get('vos_sim') or results.get('dcOp') or results.get('dc')
        noise_results = results.get('noise_sim') or results.get('noise')
        xf_resultsdict = results.get('xf_sim')
        thd_results = results.get('thd_sim')
        slew_results = results.get('slew_sim') 

        # --- Initialize Specs to None ---
        vos = None
        gain_ol = None
        gain_ol_lin = None
        gain_cl = None
        ugbw = None
        phm = None
        gm = None
        area = None
        power = None
        cmrr = None
        psrr = None

        linearity = None
        output_voltage_swing = None
        integrated_noise = None
        slew_rate = None
        settle_time = None
        thd = None
        
        valid = False


        # --- Metric Extraction ---
        ids_MM = {}
        gm_MM = {}
        vgs_MM = {}
        vds_MM = {}
        region_MM = {}

        # 1. DC Processing
        if dc_results:
            vcm = dc_results.get("cm", 0.0)
            
            # Identify VDD if possible (check params or DC results)
            vdd_val = 0.0
            if params and 'vdd' in params:
                 vdd_val = float(params['vdd'])
            elif results.get('vdd'):
                 vdd_val = float(results['vdd'])
            
            # Extract Supply Current and Compute Power
            if 'V0:p' in dc_results:
                 i_supply = np.abs(dc_results['V0:p']) 
                 if vdd_val > 0:
                      power = i_supply * vdd_val
                 else:
                      power = i_supply
            
            for comp, val in dc_results.items():
                if comp.startswith("MM") and comp.endswith("ids"):
                    ids_MM[comp.split(':')[0]] = float('%.3g' % np.abs(val))
                elif comp.startswith("MM") and comp.endswith("gm"):
                    gm_MM[comp.split(':')[0]] = float('%.3g' % np.abs(val))
                elif comp.startswith("MM") and comp.endswith("vgs"):
                    vgs_MM[comp.split(':')[0]] = float('%.3g' % np.abs(val))
                elif comp.startswith("MM") and comp.endswith("vds"):
                    vds_MM[comp.split(':')[0]] = float('%.3g' % np.abs(val))
                elif comp.startswith("MM") and comp.endswith("region"):
                    region_MM[comp.split(':')[0]] = float('%.3g' % np.abs(val))

            vos = self.find_vos(results, vcm)
            output_voltage_swing = self.find_output_voltage_swing(results, vcm)
        
        # 2. AC Processing (Single Ended)
        if ac_result_se:
            # Check for consolidated loopGain (STB analysis)
            # Search keys safely for loopGain
            keys = list(ac_result_se.keys())
            loop_key = next((k for k in keys if "loopGain" in k and "dB" not in k), None)
            
            if loop_key:
                vout_diff = ac_result_se[loop_key]
            else:
                vout_diff = None

            # Fallback: Check for loopGain_dB and phase if complex loopGain missing
            if vout_diff is None:
                 db_key = next((k for k in keys if "loopGain" in k and "dB" in k), None)
                 ph_key = next((k for k in keys if "phase" in k or "Phase" in k), None)
                 
                 if db_key and ph_key:
                      lg_db = np.array(ac_result_se[db_key])
                      lg_ph = np.array(ac_result_se[ph_key])
                      # Reconstruct complex form: 10^(dB/20) * exp(j * deg2rad(phase))
                      lg_mag = 10**(lg_db/20.0)
                      vout_diff = lg_mag * np.exp(1j * np.deg2rad(lg_ph))

            # If no loopGain, check for Voutp (Only if standard AC sweep)
            if vout_diff is None:
                 if 'Voutp' in ac_result_se:
                     vout_diff = np.array(ac_result_se['Voutp'])
                 elif 'Vout' in ac_result_se:
                     vout_diff = np.array(ac_result_se['Vout'])
                 else:
                     vout_diff = np.array([])
            else:
                 vout_diff = np.array(vout_diff) # Ensure array

            freq = ac_result_se.get('sweep_values')
            if freq is None:
                # Try finding 'freq' search key
                freq_key = next((k for k in keys if "freq" in k.lower()), None)
                if freq_key: freq = ac_result_se[freq_key]

            if len(vout_diff) > 0 and freq is not None and len(freq) > 0:
                gain_ol_lin = SpecCalc.find_dc_gain(vout_diff)
                ugbw, valid = SpecCalc.find_ugbw(freq, vout_diff)
                phm = SpecCalc.find_phm(freq, vout_diff, ugbw, valid)
                gm = SpecCalc.find_gain_margin(freq, vout_diff)
                
                # Check Linearity using DC Gain
                # linearity = self.find_linearity(results, vout_diff) 
            else:
                 # If we have Vout but no Freq, we can at least get DC gain
                 if len(vout_diff) > 0:
                      gain_ol_lin = SpecCalc.find_dc_gain(vout_diff)
                 # Add Debug info if failures persist
                 # print(f"DEBUG: Keys in ac_result: {keys}")
        
        # 2b. Area Calculation (Heuristic from Params)
        if params:
             area = 0.0
             # Schematic Area Estimation
             # nB is confirmed to be FIN COUNT.
             # We use 100nm effective width per fin.
             effective_fin_width = 100e-9 
             
             for key, val in params.items():
                  if key.startswith('nA'):
                       suffix = key[2:]
                       width_key = 'nB' + suffix
                       if width_key in params:
                            area += (val * params[width_key])

             # Convert proxy (L*N) to Physical Area (m^2)
             area *= effective_fin_width
             
             # --- Passive Component Area Estimation ---
             # Constants for Generic FinFET Node
             cap_density = 2e-3  # F/m^2 (2 fF/um^2)
             res_area_coeff = 1e-17  # m^2 per Ohm (Based on High-Res Poly)

             for key, val in params.items():
                 # Capacitors (nC...)
                 if key.startswith('nC') and val > 0:
                     area += (val / cap_density)
                 # Resistors (nR...)
                 elif key.startswith('nR') and val > 0:
                     area += (val * res_area_coeff)

        # 3. CMRR / PSRR / Closed Loop Gain Processing
        if xf_resultsdict:
             if gain_ol_lin is not None and gain_ol_lin != 0:
                  psrr = self.find_psrr(xf_resultsdict, np.abs(gain_ol_lin))
                  cmrr = self.find_cmrr_xf(xf_resultsdict, np.abs(gain_ol_lin))
             
             # Calculate Closed Loop Gain (from 'in_dc' or 'V2')
             gain_cl_lin = self.find_closed_loop_gain(xf_resultsdict)
             if gain_cl_lin is not None and gain_cl_lin != 0:
                 gain_cl = 20 * np.log10(np.abs(gain_cl_lin))

        # Convert Open Loop Gain to dB
        if gain_ol_lin is not None and gain_ol_lin != 0:
            gain_ol = 20 * np.log10(np.abs(gain_ol_lin))

        # 4. Noise Processing
        if noise_results:
             integrated_noise = SpecCalc.find_integrated_noise(noise_results)

        # 5. Transient Processing (Slew / Settling)
        if slew_results:
             time = slew_results.get('time', [])
             t_val = slew_results.get('Voutp') or slew_results.get('Vout')
             
             if (time is None or len(time) == 0) and 'sweep_values' in slew_results:
                  time = slew_results['sweep_values']

             if t_val is not None and len(t_val) > 0:
                  # Force lengths to match
                  min_len = min(len(time), len(t_val))
                  tran_data = list(zip(time[:min_len], t_val[:min_len]))
                  
                  slew_rate = SpecCalc.find_slew_rate(tran_data)
                  settle_time = SpecCalc.find_settle_time(tran_data)

        # 6. THD
        if thd_results:
             thd = self.find_thd(thd_results)

        # Return Results (Nulls preserved)
        results = dict(
            gain_ol = gain_ol,
            gain_cl = gain_cl,
            ugbw = ugbw,
            pm = phm,
            gm = gm,
            area = area,
            power = power,
            vos = vos,
            cmrr = cmrr, 
            psrr = psrr, 
            linearity = thd,
            output_voltage_swing = output_voltage_swing,
            integrated_noise = integrated_noise,
            slew_rate = slew_rate,
            settle_time = settle_time,
            valid = valid,
            zregion_of_operation_MM = region_MM,
            zzgm_MM = gm_MM,
            zzids_MM = ids_MM,
            zzvds_MM = vds_MM,
            zzvgs_MM = vgs_MM
        )
        return results

    @classmethod
    def find_psrr(self, xf_results, dc_gain_lin):
        """
        Calculates PSRR (Power Supply Rejection Ratio) in dB.
        PSRR = 20 * log10( A_dm_dc / A_supply_dc )
        Expects dc_gain_lin to be linear open loop gain.
        """
        if dc_gain_lin is None or dc_gain_lin == 0:
            return None

        keys = xf_results.keys()
        
        # Prioritize exact match for VDD source 'V0'
        vdd_key = "V0" if "V0" in keys else next((k for k in keys if "V0" in k), None)
        
        if vdd_key:
            vals = xf_results[vdd_key] 
            # Use DC value (Index 0)
            tf_vdd = float(np.abs(vals)[0])
            
            if tf_vdd > 1e-12:
                return 20 * np.log10(dc_gain_lin / tf_vdd)
            else:
                return 200.0
        return None

    @classmethod
    def find_cmrr_xf(self, xf_results, dc_gain_lin):
        """
        Calculates CMRR (Common Mode Rejection Ratio) in dB.
        """
        if dc_gain_lin is None or dc_gain_lin == 0:
            return None

        keys = xf_results.keys()
        
        # Prioritize exact match for CM source 'V1'
        cm_key = "V1" if "V1" in keys else next((k for k in keys if "V1" in k), None)
        
        # Look for the Common Mode source key
        # cm_key = next((k for k in keys if "V1" in k), None)
        
        # If we have Voutp/Voutn separate in XF results (which is common for xf (Voutp Voutn))
        # Then results might be { 'Voutp': { 'V1': ..., 'V0': ... }, 'Voutn': { ... } } 
        # OR flattened: { 'V1': [val_p, val_n]? } No, Parser structure matters.
        # Parser splits signals.
        # For 'xf_sim', it likely returns 'Voutp' and 'Voutn' keys IF they are the Output variables.
        # BUT XF returns TF *from* sources *to* the output specified in analysis.
        # If analysis is `xf_sim (Voutp Voutn) ...` it computes transf. to Voutp AND Voutn.
        # Parser usually flattens this.
        # If parsed as: data['xf_sim'] = { 'V1': ..., 'V0': ..., 'sweep_values': ... } if only 1 output?
        # But here we have two outputs.
        # libpsf behavior: if multiple outputs, it might be tricky.
        
        if cm_key:
            vals = xf_results[cm_key]
            tf_cm = float(np.abs(vals)[0])
            
            if tf_cm > 1e-12:
                return 20 * np.log10(dc_gain_lin / tf_cm)
            else:
                return 200.0
        return None

    @classmethod
    def find_closed_loop_gain(self, xf_results):
        """
        Finds Closed Loop Gain from XF analysis.
        Target source is 'V2' (or 'in_dc').
        """
        keys = xf_results.keys()
        # Look for the input source key
        input_key = next((k for k in keys if "V2" in k or "in_dc" in k), None)
        
        if input_key:
            vals = xf_results[input_key] 
            # XF vals are transfer function magnitudes
            return float(np.abs(vals)[0])
        return None

    @classmethod
    def find_thd(self, thd_results):
        """
        Calculates Total Harmonic Distortion (THD) from transient simulation.
        Returns THD in dBc.
        """
        time = thd_results.get('time', [])
        # Handle case where keys might be just lists without 'time' key if simple dict
        if (time is None or len(time) == 0) and 'sweep_values' in thd_results:
             time = thd_results['sweep_values']
             
        # Extract Single Ended Output
        v_sig = thd_results.get('Voutp') or thd_results.get('Vout')
        
        if v_sig is None or len(v_sig) == 0:
             return None

        v_sig = np.array(v_sig)
             
        if len(time) != len(v_sig):
             # Try to recover if lengths slightly mismatch due to artifacts
             min_len = min(len(time), len(v_sig))
             time = time[:min_len]
             v_sig = v_sig[:min_len]
        
        # FFT
        n = len(v_sig)
        if n < 2: return None
        
        # Windowing (Hanning) to reduce leakage
        window = np.hanning(n)
        v_windowed = v_sig * window
        
        fft_vals = np.fft.rfft(v_windowed)
        fft_mag = np.abs(fft_vals) / n
        
        # Remove DC
        fft_mag[0] = 0
        
        # Find Fundamental
        fund_idx = np.argmax(fft_mag)
        fund_mag = fft_mag[fund_idx]
        
        if fund_mag < 1e-9: return None # No signal
        
        # Sum Harmonics (2nd to 10th)
        harm_power = 0
        for i in range(2, 11):
             h_idx = fund_idx * i
             # Allow small window around exact harmonic index
             if h_idx < len(fft_mag):
                  # Simple peak finding around expected index
                  # search range +/- 2 bins
                  start = max(0, h_idx - 2)
                  end = min(len(fft_mag), h_idx + 3)
                  h_mag = np.max(fft_mag[start:end])
                  harm_power += h_mag**2
        
        thd_lin = np.sqrt(harm_power) / fund_mag
        if thd_lin <= 0: return -100.0 # Perfect?
        
        thd_db = 20 * np.log10(thd_lin)
        return thd_db

    @classmethod
    def find_closed_loop_gain(self, xf_results):
        """
        Finds Closed Loop Gain from XF analysis.
        Target source is 'V2' (or 'in_dc').
        """
        keys = xf_results.keys()
        # Look for the input source key
        input_key = next((k for k in keys if "V2" in k or "in_dc" in k), None)
        
        if input_key:
            vals = xf_results[input_key] 
            # XF vals are transfer function magnitudes
            return float(np.abs(vals)[0])
        return None

    @classmethod
    def extract_dc_sweep(self, results):
        """
        Extracts sorted (dc_offsets, vout_diff) from DC sweep.
        Handles both consolidated 'dc_swing' swept result and split 'dcswp-XXX' files.
        """
        dc_offsets = []
        vout_diffs = []
        
        # Method 1: Consolidated Swept DC Analysis (Preferred)
        if 'dc_swing' in results:
            swing_res = results['dc_swing']
            if 'sweep_values' in swing_res and 'sweep_vars' in swing_res:
                # Assuming sweep_values is a list of tuples or list, but for 1 param it's often a list
                # sweep_values typical format in libpsf is list of floats for simple sweep
                offsets_in = swing_res['sweep_values']
                
                if 'Voutp' in swing_res and 'Voutn' in swing_res:
                    v_p = np.array(swing_res['Voutp'])
                    v_n = np.array(swing_res['Voutn'])
                    # If dimensions mismatch, flatten or check
                    # They likely match sweep_values length
                    v_diff = v_p - v_n
                    return np.array(offsets_in), v_diff
                elif 'Voutp' in swing_res:
                    return np.array(offsets_in), np.array(swing_res['Voutp'])

        # Method 2: Legacy Split Files
        for result_key in results.keys():
            if result_key.startswith('dcswp-'):
                res_dict = results[result_key]
                if 'Voutp' in res_dict and 'Voutn' in res_dict:
                    val = res_dict['Voutp'] - res_dict['Voutn']
                elif 'Voutp' in res_dict:
                     val = res_dict['Voutp']
                else: 
                     val = 0.0
                
                try:
                    num_part = result_key.split('_')[0].split('-')[1]
                    idx = int(num_part)
                    # Differential specific sweep logic
                    dc_offset = -0.1 + (idx * 0.001)
                    
                    dc_offsets.append(dc_offset)
                    vout_diffs.append(val)
                except:
                    pass

        dc_offsets = np.array(dc_offsets)
        vout_diffs = np.array(vout_diffs)
        
        if len(dc_offsets) > 0:
            sort_idx = np.argsort(dc_offsets)
            return dc_offsets[sort_idx], vout_diffs[sort_idx]
        return np.array([]), np.array([])

    @classmethod
    def find_vos(self, results, vcm):
        dc_offsets, vouts = self.extract_dc_sweep(results)
        if len(dc_offsets) < 4: return None

        spline = interp.UnivariateSpline(dc_offsets, vouts, s=0)
        def root_func(x): return float(spline(x)) - vcm
        try:
            return sciopt.brentq(root_func, dc_offsets[0], dc_offsets[-1])
        except ValueError:
            return None

    # @classmethod
    # def find_linearity(self, results, vout_diff, allowed_deviation_pct=2.0):
    #     # Similar to Diff logic but uses single ended params
    #     gain = SpecCalc.find_dc_gain(vout_diff)
    #     # Allow linearity calculation even if gain is low/None, to report the range
    #     # if gain is None or gain < 1: return None

    #     dc_offsets, vouts = self.extract_dc_sweep(results)
    #     if len(dc_offsets) < 4: return None

    #     spline = interp.UnivariateSpline(dc_offsets, vouts, s=0)
    #     slope_spline = spline.derivative(n=1)
    #     fine_x = np.linspace(dc_offsets.min(), dc_offsets.max(), 2000)
    #     fine_slope = slope_spline(fine_x)
        
    #     # Safe finding of zero crossing for linearity center
    #     zero_idxs = np.where(np.isclose(fine_x, 0, atol=1e-3))[0]
    #     if len(zero_idxs) > 0:
    #          zero_idx = zero_idxs[0]
    #     else:
    #          zero_idx = np.argmin(np.abs(fine_x))

    #     slope_at_zero = fine_slope[zero_idx]
    #     allowed_dev = abs(slope_at_zero) * (allowed_deviation_pct / 100.0)

    #     left_idx = zero_idx
    #     while left_idx > 0 and abs(fine_slope[left_idx] - slope_at_zero) <= allowed_dev:
    #         left_idx -= 1
    #     right_idx = zero_idx
    #     while right_idx < len(fine_x) - 1 and abs(fine_slope[right_idx] - slope_at_zero) <= allowed_dev:
    #         right_idx += 1

    #     return (fine_x[left_idx], fine_x[right_idx])


    @classmethod
    def find_output_voltage_swing(self, results, vcm, allowed_deviation_pct=10.0):
        dc_offsets, vouts = self.extract_dc_sweep(results)
        if len(dc_offsets) < 4: return None

        spline = interp.UnivariateSpline(dc_offsets, vouts, s=0)
        slope_spline = spline.derivative()
        
        mid_idx = np.argmin(np.abs(dc_offsets))
        max_slope = float(slope_spline(dc_offsets[mid_idx]))
        allowed_dev = abs(max_slope) * (allowed_deviation_pct / 100.0)
        
        idx_left = mid_idx
        while idx_left > 0:
             if abs(float(slope_spline(dc_offsets[idx_left])) - max_slope) > allowed_dev:
                  break
             idx_left -= 1
        idx_right = mid_idx
        while idx_right < len(dc_offsets) - 1:
             if abs(float(slope_spline(dc_offsets[idx_right])) - max_slope) > allowed_dev:
                  break
             idx_right += 1
             
        return (vouts[idx_left], vouts[idx_right])
    
    # Redundant methods removed/replaced by SpecCalc calls:
    # find_dc_gain, find_ugbw, find_phm, find_integrated_noise, find_slew_rate, find_settle_time, _get_best_crossing
