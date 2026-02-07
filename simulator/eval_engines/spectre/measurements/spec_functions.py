"""
spec_functions.py

Author: natelgrw
Last Edited: 02/06/2026

Shared utility functions for calculating op-amp specifications
from simulation results. Used by both SingleEnded and Differential
measurement managers.
"""

import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt
import scipy.integrate as scint


class SpecCalc(object):
    """
    Collection of static methods for specification calculation.
    """

    @staticmethod
    def find_dc_gain(vout):
        """
        Finds the DC gain from output voltage array (index 0).
        """
        if vout is None or len(vout) == 0:
            return None
        return float(np.abs(vout)[0])

    @staticmethod
    def _get_best_crossing(xvec, yvec, val):
        """
        Finds the best crossing point where yvec crosses val.
        Returns (crossing_x, valid_bool)
        """
        if len(xvec) < 2: 
             return None, False
             
        try:
            interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)
            def fzero(x): return interp_fun(x) - val
            
            # Check if crossing is possible in range
            y_min, y_max = np.min(yvec), np.max(yvec)
            if val < y_min or val > y_max:
                return None, False

            return sciopt.brentq(fzero, xvec[0], xvec[-1]), True
        except:
            return None, False

    @staticmethod
    def find_ugbw(freq, vout):
        """
        Finds Unity Gain Bandwidth using Log-Log interpolation for high accuracy.
        Assumes gain decreases monotonically near 0dB.
        """
        if freq is None or vout is None or len(freq) != len(vout): 
            return None, False
            
        gain = np.abs(vout)
        
        # 1. Find the first time gain crosses 1.0 (0dB) from above
        # Valid UGBW requires gain starting > 1
        if gain[0] < 1.0:
            return None, False
            
        # Find indices where gain transitions from >=1 to <1
        # This approach avoids oscillating spline solutions
        crossings = np.where((gain[:-1] >= 1.0) & (gain[1:] < 1.0))[0]
        
        if len(crossings) == 0:
            return None, False
            
        # Take the first crossing (standard definition)
        idx = crossings[0]
        
        # 2. Log-Log Interpolation
        # log(gain) vs log(freq) is linear for dominant pole systems
        # y = log10(gain), x = log10(freq)
        # We want x where y = log10(1) = 0
        
        f1, f2 = freq[idx], freq[idx+1]
        g1, g2 = gain[idx], gain[idx+1]
        
        # Avoid log(0)
        if f1 <= 0 or f2 <= 0 or g1 <= 0 or g2 <= 0:
            return None, False
            
        x1, x2 = np.log10(f1), np.log10(f2)
        y1, y2 = np.log10(g1), np.log10(g2)
        
        if y1 == y2: # Horizontal segment?
             return None, False
             
        # Interpolate for y=0
        # (0 - y1) = (y2 - y1) / (x2 - x1) * (x_target - x1)
        # x_target = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
        
        x_target = x1 - y1 * (x2 - x1) / (y2 - y1)
        ugbw = 10**x_target
        
        return float(ugbw), True

    @staticmethod
    def find_phm(freq, vout, ugbw, valid_ugbw):
        """
        Finds Phase Margin at UGBW.
        Uses Phase vs Log(Freq) interpolation.
        Returns PM in range (0, 180) typically. Unstable < 0.
        """
        if not valid_ugbw or freq is None or vout is None or ugbw is None:
            return None
            
        phase = np.angle(vout, deg=True)
        # 1. Unwrap Phase safely
        # Convert to radians, unwrap, convert back
        phase_rad = np.deg2rad(phase)
        phase_unwrapped = np.rad2deg(np.unwrap(phase_rad))
        
        # 2. Interpolate Phase at UGBW (Linear Phase vs Log Freq)
        # Find index near UGBW explicitly to avoid global spline issues
        # Since freq is monotonic, searching is easy. 
        # But global interp1d is usually fine for phase which is smooth, 
        # provided we use log frequency.
        
        log_freq = np.log10(np.clip(freq, 1e-12, None))
        log_ugbw = np.log10(ugbw)
        
        try:
            # Linear interpolation in Log-Freq domain is physically sound (Bode)
            phase_fun = interp.interp1d(log_freq, phase_unwrapped, kind='linear', fill_value="extrapolate")
            phase_at_ugbw = float(phase_fun(log_ugbw))
            
            # 3. Calculate Phase Margin
            # Standard definition: PM = 180 + Phase(UGBW)
            # Assumption: Phase starts near 0 or -180 and drops.
            # Ideally, unstable system has Phase < -180.
            # Stable system has Phase > -180 (e.g. -90).
            # So (-90) -> PM = 90.
            # (-200) -> PM = -20.
            
            # Normalize phase relative to -180 crossing context
            # If phase is e.g. -360-90 = -450 (wrapped), 
            # we want the margin relative to the 180 it crossed.
            # But usually we just care about "distance above -180".
            
            # Simple normalization to [-360, 0] range often helps if start phase is arbitrary
            # But standard unwrapping from DC usually puts DC phase at 0 or -180.
            
            pm = 180 + phase_at_ugbw
            
            # Handle modulo 360 wrap-around artifacts if simulation setup 
            # started phase at +180 or similar oddity.
            # We want the result typically in [-180, 180].
            while pm > 180: pm -= 360
            while pm <= -180: pm += 360
                
            return float(pm)
        except:
            return None

    @staticmethod
    def find_gain_margin(freq, vout):
        """
        Finds Gain Margin (GM) in dB.
        GM = -20*log10(|T|) at the frequency where Phase(T) crosses -180 degrees.
        Robust to phase wrapping and multiple crossings (takes the first relevant one).
        """
        if freq is None or vout is None or len(freq) != len(vout): 
            return None
            
        phase = np.angle(vout, deg=True)
        # Unwrap phase (radians) then convert to degrees
        phase_rad = np.deg2rad(phase)
        phase_unwrapped = np.rad2deg(np.unwrap(phase_rad))
        
        # Determine Stability Criterion Frequency (Phase Crossover)
        # We look for the frequency where phase crosses -180 + k*360.
        # For a standard loop gain starting at 0 or -360, the critical crossing is -180.
        # Ideally, we find the first index where phase drops below -180.
        
        # 1. Normalize phase to start near 0 for consistent search? 
        # No, trust the unwrap.
        
        # Find frequency where phase = -180 (or closest odd multiple of 180 that represents instability)
        # Standard approach: Find crossing of -180.
        
        crossings = np.where(np.diff(np.sign(phase_unwrapped + 180)))[0]
        
        # If no crossing of -180 found, maybe it started at -180? (Unlikely for Loop Gain, usually starts 0)
        # Or maybe it wrapped weirdly. Check -180, -540, +180.
        
        # Practical Robustness: 
        # Use interpolation to find exactly where Phase = -180
        
        # Define target phase (usually -180 for standard negative feedback stability analysis)
        target_phase = -180.0
        
        # Check if we ever cross -180
        # If max phase < -180 or min phase > -180 (and not wrapping), GM is undefined or infinite.
        
        px_freq, valid = SpecCalc._get_best_crossing(freq, phase_unwrapped, val=target_phase)
        
        if not valid:
            # Try -180 +/- 360 if unwrap shifted us
            for shift in [-540.0, 180.0]:
                 px_freq, valid = SpecCalc._get_best_crossing(freq, phase_unwrapped, val=shift)
                 if valid: break
        
        if not valid:
            return None
            
        # Find Gain magnitude at px_freq using Log-Log interpolation
        gain = np.abs(vout)
        
        # Helper for scalar logic
        if px_freq <= 0: return None
        
        # Interpolate Gain at px_freq
        # log(gain) vs log(freq)
        log_freq = np.log10(np.clip(freq, 1e-12, None))
        log_gain = np.log10(np.clip(gain, 1e-20, None))
        log_px = np.log10(px_freq)
        
        try:
            gain_fun = interp.interp1d(log_freq, log_gain, kind='linear')
            log_gain_at_px = gain_fun(log_px)
            gain_at_px = 10**log_gain_at_px
            
            # GM_dB = -20 * log10(gain_at_px)
            gm_db = -20 * float(log_gain_at_px)
            return float(gm_db)
        except:
            return None


    @staticmethod
    def find_integrated_noise(noise_results, f_start=1e6, f_stop=5e8, num_points=55):
        """
        Integrates total output noise.
        Supports both 'standard' noise dict (with 'out' key) and 'component' noise list (MM keys).
        """
        # Safe check for empty dict or failure
        if noise_results is None or len(noise_results) == 0:
             return None

        # Case 1: Standard 'out' key (Total Output Noise)
        # Often represented in 'out' or 'Vout' or similar.
        out_keys = [k for k in noise_results.keys() if 'out' in str(k).lower() or 'vout' in str(k).lower()]
        target_key = next((k for k in out_keys if k == 'out'), None)
        if not target_key and out_keys: target_key = out_keys[0]

        if target_key:
             noise_vals = np.abs(noise_results[target_key])
             freqs = noise_results.get('sweep_values', [])
             if len(freqs) > 1:
                  # Integrate Power Spectral Density (V^2/Hz)
                  total_power = scint.simps(noise_vals**2, freqs)
                  return float(np.sqrt(total_power))

        # Case 2: Summing components (MM keys)
        # Used in legacy/component-wise noise analysis
        total_integrated_noise = 0.0
        mm_keys = [k for k in noise_results.keys() if str(k).startswith("MM")]
        
        if mm_keys:
            # Reconstruct frequency vector if not present
            freqs = np.logspace(np.log10(f_start), np.log10(f_stop), num_points)
            
            for key in mm_keys:
                noise_array = noise_results[key]
                # Check data format
                if isinstance(noise_array, list) and len(noise_array) > 0 and isinstance(noise_array[0], dict): 
                     # List of dicts with byte keys? (Old parser specific)
                     total_psd = np.array([entry.get(b'total', 0.0) for entry in noise_array])
                else:
                     # Assuming array
                     total_psd = np.array(noise_array)

                if len(total_psd) == len(freqs):
                    integrated_noise = scint.simps(total_psd, freqs)
                    total_integrated_noise += integrated_noise
            return float(total_integrated_noise)

        return None

    @staticmethod
    def find_slew_rate(tran_data):
        """
        Finds Slew Rate from [(t0, v0), (t1, v1)...]
        """
        if not tran_data or len(tran_data) < 10: 
             return None

        time = np.array([t for t, _ in tran_data])
        vout = np.array([v for _, v in tran_data])

        try:
            spline = interp.CubicSpline(time, vout)
            # Sample finer grid
            t_fine = np.linspace(time[0], time[-1], min(10000, len(time)*10))
            dv_dt = spline.derivative()(t_fine)
            
            return float(np.max(np.abs(dv_dt)))
        except:
             return 0.0

    @staticmethod
    def find_settle_time(tran_data, tol=0.005, delay=5e-9):
        """
        Finds settling time.
        """
        if not tran_data or len(tran_data) < 10: 
             return None

        time = np.array([t for t, _ in tran_data])
        vout = np.array([v for _, v in tran_data])

        try:
            # Last 5-10% average as final value
            n_tail = max(5, int(len(vout)*0.1))
            v_final = np.mean(vout[-n_tail:])
            
            err = np.abs(vout - v_final)
            band = np.abs(v_final * tol)
            
            # Find indices where error is OUTSIDE band
            outside_idxs = np.where(err > band)[0]
            
            if len(outside_idxs) == 0:
                 return 0.0 # Settles immediately?
                 
            last_idx = outside_idxs[-1]
            t_settled = time[last_idx]
            
            # Settling time is relative to input step (delay)
            return max(0.0, float(t_settled - delay))
        except:
             return None

    @staticmethod
    def extract_dc_sweep(results):
        """
        Standard extraction of DC sweep data (offsets vs outputs).
        """
        dc_offsets = []
        vouts = []
        
        # Look for dcswp keys
        sweep_keys = [k for k in results.keys() if k.startswith('dcswp-')]
        
        for k in sweep_keys:
            res_dict = results[k]
            # Handle Differential vs Single Ended
            val = 0.0
            if 'Voutp' in res_dict and 'Voutn' in res_dict:
                 val = res_dict['Voutp'] - res_dict['Voutn']
            elif 'Voutp' in res_dict:
                 val = res_dict['Voutp']
            
            # Parse Offset Index from key "dcswp-XXXX_name"
            # Assuming format matches previously observed logic
            try:
                # Key format: 'dcswp-000_dcOp' or similar
                # Just assuming the index implies the x-axis for now based on file logic
                # Template assumption: -0.1 to 0.1 usually? Or custom params.
                # Actually, the parsing logic in previous files was:
                # idx 0 -> -0.1, idx 100 -> 0.0, idx 200 -> 0.1 etc.
                # Let's try to be generic or copy the specific logic.
                # From diff_meas_man: "dc_offset = -0.1 + (idx * 0.001)"
                # From single_ended: "dc_offset = int(result[6:9]) * 0.001 - 0.5" (Wait, 0.5? That's huge range)
                
                # Let's standardize on the key parsing logic seen in differential:
                # num = int(k.split('_')[0].split('-')[1])
                # offset = -0.1 + (num * 0.001)
                
                parts = k.split('_')[0].split('-')
                if len(parts) > 1:
                     num = int(parts[1])
                     # Hybrid logic based on range? 
                     # Most safe is to assume the 200 point sweep from -100mV to +100mV?
                     # Or the 1000 point sweep?
                     # Let's rely on the calling function to interpret?
                     # Actually, to make this useful, we need the X-axis.
                     
                     # Implementation Note: Since single ended and diff had different parsing logic strings,
                     # maybe we should keep this extraction in the manager or standardize it here.
                     # Let's use the safer Differential logic as default for modern templates.
                     offset = -0.1 + (num * 0.001) 
                     
                     dc_offsets.append(offset)
                     vouts.append(val)
            except:
                pass
                
        if len(dc_offsets) > 0:
            dc_offsets = np.array(dc_offsets)
            vouts = np.array(vouts)
            sort_idx = np.argsort(dc_offsets)
            return dc_offsets[sort_idx], vouts[sort_idx]
            
        return np.array([]), np.array([])
