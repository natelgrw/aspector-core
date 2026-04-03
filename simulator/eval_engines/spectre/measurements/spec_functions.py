"""
spec_functions.py

Author: natelgrw
Last Edited: 04/01/2026

Shared utility functions for calculating op-amp specifications
from simulation results. Used by both single ended and differential
measurement managers.
"""

import numpy as np
import scipy.integrate as sinteg
import scipy.interpolate as interp
import scipy.optimize as sciopt


class SpecCalc:
    """
    Static helper functions for measurement and spec calculations.
    Supported specs for differential and single ended op-amps include:

    - Open Loop DC Gain (dB)
    - Unity-Gain Bandwidth (Hz)
    - Phase Margin (degrees)
    - DC CMRR (dB)
    - DC PSRR (dB)
    - Power (W)
    - Estimated Area (um^2)
    - Integrated Noise (Vrms)
    - Settling Time (ns)
    - Slew Rate (V/us)
    - Vos (V)
    - Output Swing (V, V)
    - Total Harmonic Distortion (dB)
    """

    @staticmethod
    def _coerce_signal_vector(signal):
        """
        Normalize libpsf signal containers into a flat numeric vector.

        Parameters:
        -----------
        signal: The raw signal data from libpsf, which can be in various formats

        Returns:
        --------
        np.ndarray: A 1D numpy array of floats or complex numbers representing the signal.
        """
        if signal is None:
            return None

        arr = np.asarray(signal)
        if arr.ndim == 0:
            return arr.reshape(1)

        if np.iscomplexobj(arr):
            return np.ravel(arr.astype(complex, copy=False))

        if arr.ndim >= 2 and arr.shape[-1] == 2:
            flat = np.reshape(arr, (-1, 2))
            return flat[:, 0].astype(float) + 1j * flat[:, 1].astype(float)

        if arr.dtype != object:
            return np.ravel(arr)

        coerced = []
        for sample in signal:
            if isinstance(sample, (list, tuple, np.ndarray)) and len(sample) == 2:
                coerced.append(complex(float(sample[0]), float(sample[1])))
            else:
                coerced.append(sample)
        return np.asarray(coerced)

    @staticmethod
    def find_dc_gain(vout):
        """
        Finds the DC gain from output voltage array (index 0).

        Parameters:
        -----------
        vout (array-like): The output voltage samples from the simulation.

        Returns:
        --------
        float: The estimated DC gain in V/V, or None if it cannot be determined.
        """
        if vout is None or len(vout) == 0:
            return None
        return float(np.abs(vout)[0])

    @staticmethod
    def _get_best_crossing(xvec, yvec, val, use_log_x=False):
        """
        Find first crossing point with linear interpolation.

        Parameters:
        -----------
        xvec (array-like): The x-axis values (e.g., frequency or time).
        yvec (array-like): The y-axis values (e.g., gain or voltage).
        val (float): The target y-value to find the crossing for.
        use_log_x (bool): If True, performs interpolation in log-x space.

        Returns:
        --------
        tuple: (crossing_x, valid) where crossing_x is the estimated x-value of the crossing, and valid is a 
               boolean indicating if a crossing was found.

        """
        if len(xvec) < 2 or len(yvec) < 2:
            return None, False

        try:
            x_arr = np.ravel(np.asarray(xvec, dtype=float))
            y_arr = np.ravel(np.asarray(yvec, dtype=float))
        except Exception:
            return None, False

        n = min(len(x_arr), len(y_arr))
        if n < 2:
            return None, False
        x_arr = x_arr[:n]
        y_arr = y_arr[:n]

        finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_arr = x_arr[finite_mask]
        y_arr = y_arr[finite_mask]
        if len(x_arr) < 2:
            return None, False

        # find first interval where y-val changes sign
        y_shifted = y_arr - val
        sign_changes = np.where(np.diff(np.sign(y_shifted)) != 0)[0]

        if len(sign_changes) == 0:
            return None, False

        # take first crossing
        idx = sign_changes[0]
        
        x1, x2 = x_arr[idx], x_arr[idx+1]
        y1, y2 = y_arr[idx], y_arr[idx+1]

        # linear interpolation
        if np.isclose(y2, y1):
            return float(x1), True

        if use_log_x:
            log_x1, log_x2 = np.log10(max(x1, 1e-15)), np.log10(max(x2, 1e-15))
            log_x_target = log_x1 + (val - y1) * (log_x2 - log_x1) / (y2 - y1)
            return 10**log_x_target, True
        else:
            x_target = x1 + (val - y1) * (x2 - x1) / (y2 - y1)
            return float(x_target), True

    @staticmethod
    def find_ugbw(freq, vout):
        """
        Find unity-gain bandwidth using log-log interpolation.

        Parameters:
        -----------
        freq (array-like): The frequency samples from the simulation.
        vout (array-like): The complex output voltage samples.

        Returns:
        --------
        tuple: (ugbw, valid)
        """
        if freq is None or vout is None or len(freq) < 2 or len(freq) != len(vout): 
            return None, False
            
        gain = np.abs(vout)
        
        # valid ugbw requires initial gain above unity
        if gain[0] < 1.0:
            return None, False
            
        # find first gain crossing from above unity to below unity
        crossings = np.where((gain[:-1] >= 1.0) & (gain[1:] < 1.0))[0]
        
        if len(crossings) == 0:
            return None, False
            
        idx = crossings[0]
        
        # log-log interpolation
        f1, f2 = max(freq[idx], 1e-12), max(freq[idx+1], 1e-12)
        g1, g2 = max(gain[idx], 1e-12), max(gain[idx+1], 1e-12)
        
        log_f1, log_f2 = np.log10(f1), np.log10(f2)
        log_g1, log_g2 = np.log10(g1), np.log10(g2)
        
        if np.isclose(log_g1, log_g2): 
             log_f_target = log_f1
        else:
             log_f_target = log_f1 - log_g1 * (log_f2 - log_f1) / (log_g2 - log_g1)
             
        ugbw = 10**log_f_target
        
        return float(ugbw), True

    @staticmethod
    def find_phm(freq, vout, ugbw, valid_ugbw):
        """
        Find phase margin at unity-gain bandwidth (or any target frequency).
        Uses fast local interpolation, binary search, and strictly preserves initial phase lag.
        """
        if not valid_ugbw or freq is None or vout is None or ugbw is None or np.isnan(ugbw) or ugbw <= 0:
            return None

        try:
            freq_arr = np.ravel(np.asarray(freq, dtype=float))
            vout_arr = np.ravel(np.asarray(vout))
        except Exception:
            return None

        n = min(len(freq_arr), len(vout_arr))
        if n < 2:
            return None
        freq_arr = freq_arr[:n]
        vout_arr = vout_arr[:n]

        finite_mask = np.isfinite(freq_arr) & np.isfinite(np.abs(vout_arr))
        freq_arr = freq_arr[finite_mask]
        vout_arr = vout_arr[finite_mask]
        if len(freq_arr) < 2:
            return None

        order = np.argsort(freq_arr)
        freq_arr = freq_arr[order]
        vout_arr = vout_arr[order]

        uniq_freq, uniq_idx = np.unique(freq_arr, return_index=True)
        freq_arr = uniq_freq
        vout_arr = vout_arr[uniq_idx]
        if len(freq_arr) < 2:
            return None
            
        # S-Tier Fix: Binary search for the index
        insert_idx = np.searchsorted(freq_arr, ugbw)
        
        # Guard against edge cases (extrapolation out of bounds)
        if insert_idx == 0 or insert_idx >= len(freq_arr):
            return None
            
        # The crossing is between idx and idx+1
        idx = insert_idx - 1
        
        # Unwrap phase to prevent artificial jump artifacts
        phase_unwrapped = np.unwrap(np.angle(vout_arr))
        
        p1, p2 = phase_unwrapped[idx], phase_unwrapped[idx+1]
        f1, f2 = freq_arr[idx], freq_arr[idx+1]
        
        log_f1, log_f2 = np.log10(max(f1, 1e-12)), np.log10(max(f2, 1e-12))
        log_ugbw = np.log10(ugbw)
        
        # Local linear interpolation of the unwrapped phase
        if np.isclose(log_f1, log_f2):
            p_target = p1
        else:
            p_target = p1 + (log_ugbw - log_f1) * (p2 - p1) / (log_f2 - log_f1)
            
        p_target_deg = np.rad2deg(p_target)
        
        # Start exactly from the simulated phase lag at freq[0]. No "ideal" snapping.
        dc_phase_actual_deg = np.rad2deg(phase_unwrapped[0])
        
        # Phase shift relative to the ACTUAL simulated baseline
        phase_shift_deg = p_target_deg - dc_phase_actual_deg
        
        # Strict Phase Margin. NO modulo wrapping.
        pm = 180.0 + phase_shift_deg
        
        return float(pm)

    @staticmethod
    def find_integrated_noise(noise_results, f_start=None, f_stop=None):
        """
        Integrate total output noise over a specified frequency band.
        Strictly requires the simulator to provide a unified output noise vector (ASD)
        to guarantee correct correlated noise cancellation (e.g., tail current noise).

        Parameters:
        -----------
        noise_results (dict): The noise simulation results.
        f_start (float): The start frequency for integration.
        f_stop (float): The stop frequency for integration.

        Returns:
        --------
        float: The integrated noise in Vrms, or None if not valid.
        """
        if noise_results is None or len(noise_results) == 0:
            return None

        freqs = np.ravel(np.asarray(noise_results.get('sweep_values', []), dtype=float))
        if len(freqs) < 2:
            return None

        # S-Tier Fix: Enforce frequency bounds so 1/f noise doesn't blow up the integral
        start_idx = 0
        stop_idx = len(freqs)
        
        if f_start is not None:
            insert_start = np.searchsorted(freqs, f_start)
            if insert_start < len(freqs):
                start_idx = insert_start
                
        if f_stop is not None:
            insert_stop = np.searchsorted(freqs, f_stop)
            if insert_stop <= len(freqs):
                stop_idx = insert_stop
            
        # Ensure valid slice
        if start_idx >= stop_idx or start_idx >= len(freqs):
            return None
            
        freqs_sliced = freqs[start_idx:stop_idx]

        # Strict Mode: Only accept the unified 'out' vector. 
        if 'out' not in noise_results:
            return None

        # ASD is V/sqrt(Hz), square before integration to get V^2/Hz
        try:
            noise_asd = np.ravel(np.abs(np.asarray(noise_results['out'], dtype=float)))
        except Exception:
            return None
            
        # Slice to match frequency bounds
        n_len = min(len(freqs), len(noise_asd))
        if start_idx >= n_len:
            return None
            
        noise_asd_sliced = noise_asd[start_idx:min(stop_idx, n_len)]
        
        if len(noise_asd_sliced) < 2:
            return None
        
        # Integrate Power Spectral Density
        total_power_v2 = sinteg.trapezoid(noise_asd_sliced**2, freqs_sliced[:len(noise_asd_sliced)])
        return float(np.sqrt(max(total_power_v2, 0.0)))


    @staticmethod
    def find_slew_rate(tran_data, delay=5.0, lo_pct=0.1, hi_pct=0.9, min_swing=10.0):
        """
        Find slew rate from transient waveform.
        Uses SciPy PCHIP interpolation over dense dynamic time-steps.

        Units: time [ns], vout [mV]. Returns [V/us].
        """
        if not tran_data or len(tran_data) < 10:
            return None

        # Convert to numpy arrays
        try:
            data = np.asarray(tran_data, dtype=float)
        except Exception:
            return None
        if data.ndim != 2 or data.shape[1] < 2:
            return None
        time, vout = data[:, 0], data[:, 1]

        # 1. Isolate post-step region
        mask = time >= delay
        if np.sum(mask) < 5: 
            return None
            
        t_s, v_s = time[mask], vout[mask]

        # S-TIER FIX: The SciPy Scrubber. 
        # Removes duplicate timestamps caused by Spectre solver resets.
        t_s, unique_indices = np.unique(t_s, return_index=True)
        v_s = v_s[unique_indices]
        
        if len(t_s) < 5:
            return None

        # 2. Estimate steady-state from tail window
        t_end = t_s[-1]
        final_window_mask = t_s >= (t_end - 2.0) 
        
        v_init = v_s[0]
        v_final = np.mean(v_s[final_window_mask]) if np.any(final_window_mask) else v_s[-1]
        delta_v = v_final - v_init

        # Reject small/railed movement
        if np.abs(delta_v) < min_swing:
            return None 

        # 3. Define 10/90 thresholds
        v10 = v_init + lo_pct * delta_v
        v90 = v_init + hi_pct * delta_v

        # 4. Interpolate with PCHIP
        try:
            v_interp = interp.PchipInterpolator(t_s, v_s, extrapolate=False)
        except Exception:
            return None

        def _first_crossing_time(threshold, rising):
            # Bracket first crossing
            if rising:
                bracket = np.where((v_s[:-1] <= threshold) & (v_s[1:] > threshold))[0]
            else:
                bracket = np.where((v_s[:-1] >= threshold) & (v_s[1:] < threshold))[0]

            if len(bracket) == 0:
                return None

            i = int(bracket[0])
            t0, t1 = float(t_s[i]), float(t_s[i + 1])

            try:
                # Root-find on PCHIP curve
                return float(sciopt.brentq(lambda tt: float(v_interp(tt) - threshold), t0, t1))
            except Exception:
                # Linear fallback if BrentQ fails to converge
                v0, v1 = float(v_s[i]), float(v_s[i + 1])
                return float(t0 + (t1 - t0) * (threshold - v0) / (v1 - v0))

        is_rising = bool(delta_v > 0)
        t10 = _first_crossing_time(v10, rising=is_rising)
        t90 = _first_crossing_time(v90, rising=is_rising)

        if t10 is None or t90 is None:
            return None

        dt = float(t90 - t10)
        dv = float(v90 - v10)

        # 5. Resolution guard
        points_in_transition = np.sum((t_s >= t10) & (t_s <= t90))
        if points_in_transition < 3:
            return None

        # 6. Final calculation (mV/ns is numerically equivalent to V/us)
        if dt < 1e-12: 
            return None
            
        return float(np.abs(dv / dt))

    @staticmethod
    def find_settle_time(tran_data, vdd, tol=0.01, step_fraction=0.05, delay=5.0, noise_floor_mv=0.1):
        """
        Find settling time from transient waveform using SciPy PCHIP.
        """
        if not tran_data or len(tran_data) < 10:
            return None

        try:
            data = np.asarray(tran_data, dtype=float)
        except Exception:
            return None
        if data.ndim != 2 or data.shape[1] < 2:
            return None
        t, v = data[:, 0], data[:, 1]

        post_step_mask = t >= delay
        if np.sum(post_step_mask) < 10:
            return None

        t_post, v_post = t[post_step_mask], v[post_step_mask]

        # S-TIER FIX: The SciPy Scrubber
        t_post, unique_indices = np.unique(t_post, return_index=True)
        v_post = v_post[unique_indices]

        # 1. Establish tolerance band from stimulus
        ideal_step_size = float(vdd) * step_fraction
        band = max(ideal_step_size * tol, noise_floor_mv)
        
        # 2. Estimate final steady-state
        tail_idx = max(5, int(len(v_post) * 0.1))
        v_final = np.mean(v_post[-tail_idx:])

        # 3. Reject dead/railed designs
        pre_step_mask = (t < delay) & (t > delay - 1.0) 
        v_initial = np.mean(v[pre_step_mask]) if np.sum(pre_step_mask) > 0 else v_post[0]
        achieved_swing = np.abs(v_final - v_initial)
        
        if achieved_swing < (ideal_step_size * 0.5):
            return None

        # 4. Settle check
        err = np.abs(v_post - v_final)
        unsettled_indices = np.where(err > band)[0]
        
        if len(unsettled_indices) == 0:
            return 0.0
            
        last_unsettled_idx = unsettled_indices[-1]
        
        if last_unsettled_idx >= len(t_post) - 2:
            return None
            
        # 5. Precise crossing extraction
        i0, i1 = int(last_unsettled_idx), int(last_unsettled_idx + 1)
        t0, t1 = float(t_post[i0]), float(t_post[i1])
        e0, e1 = float(err[i0] - band), float(err[i1] - band)

        t_settled = t1
        try:
            v_interp = interp.PchipInterpolator(t_post, v_post, extrapolate=False)
            def ferr(tt):
                return abs(float(v_interp(tt)) - float(v_final)) - float(band)

            if e0 > 0.0 and e1 <= 0.0:
                t_settled = float(sciopt.brentq(ferr, t0, t1))
        except Exception:
            # Linear fallback
            if e0 > 0.0 and e1 <= 0.0 and not np.isclose(e0, e1):
                t_settled = float(t0 + (t1 - t0) * (-e0) / (e1 - e0))
        
        return float(max(0.0, t_settled - delay))

    @staticmethod
    def extract_dc_sweep(results):
        """
        Extract DC sweep data as sorted 1D arrays (dc_offset, output).
        Uses `swing_sweep` and prefers differential output when available.
        """
        def _to_1d_float(vec):
            arr = np.asarray(vec)
            if arr.ndim > 1:
                arr = np.ravel(arr)
            return arr.astype(float)

        def _sanitize_xy(x, y):
            try:
                x = _to_1d_float(x)
                y = _to_1d_float(y)
            except Exception:
                return np.array([]), np.array([])

            n = min(len(x), len(y))
            if n < 2:
                return np.array([]), np.array([])
            x = x[:n]
            y = y[:n]

            finite = np.isfinite(x) & np.isfinite(y)
            x = x[finite]
            y = y[finite]
            if len(x) < 2:
                return np.array([]), np.array([])

            order = np.argsort(x)
            x = x[order]
            y = y[order]

            # merge duplicate x-values by averaging y-values
            uniq_x, inv = np.unique(x, return_inverse=True)
            if len(uniq_x) != len(x):
                y_accum = np.bincount(inv, weights=y)
                y_count = np.bincount(inv)
                y = y_accum / np.maximum(y_count, 1)
                x = uniq_x

            return x, y

        if not isinstance(results, dict):
            return np.array([]), np.array([])

        # locate swing sweep dict robustly: exact key first, then prefix variants
        swing_res = None
        if 'swing_sweep' in results and isinstance(results.get('swing_sweep'), dict):
            swing_res = results['swing_sweep']
        else:
            for k, v in results.items():
                if isinstance(k, str) and k.startswith('swing_sweep') and isinstance(v, dict):
                    swing_res = v
                    break

        if isinstance(swing_res, dict) and 'sweep_values' in swing_res:
            x_raw = swing_res['sweep_values']

            # support parser/libpsf signal-name case variants
            signal_lookup = {str(k): k for k in swing_res.keys()}
            voutp_key = signal_lookup.get('Voutp')
            voutn_key = signal_lookup.get('Voutn')

            if voutp_key is not None and voutn_key is not None:
                y_raw = np.asarray(swing_res[voutp_key]) - np.asarray(swing_res[voutn_key])
            elif voutp_key is not None:
                y_raw = swing_res[voutp_key]
            else:
                return np.array([]), np.array([])

            x, y = _sanitize_xy(x_raw, y_raw)
            if len(x) >= 2:
                return x, y

        return np.array([]), np.array([])

    @staticmethod
    def find_estimated_area(params):
        """
        Estimate total on-chip area in um^2 from sizing parameters.
        """
        if not isinstance(params, dict) or len(params) == 0:
            return None

        # node-specific design rules (cpp: contacted poly pitch, pitch: fin/metal pitch)
        # cap density in f/um^2, res coeff in um^2/ohm
        node_map = {
            7: {
                'LSTP': {'cpp': 54e-9, 'pitch': 27e-9, 'cap_density': 2.0e-15 / 1e-12, 'res_coeff': 400},
                'HP':   {'cpp': 57e-9, 'pitch': 30e-9, 'cap_density': 2.0e-15 / 1e-12, 'res_coeff': 400}
                # source: ASAP7, AMD
            },
            10: {
                'LSTP': {'cpp': 54e-9, 'pitch': 34e-9, 'cap_density': 1.8e-15 / 1e-12, 'res_coeff': 300},
                'HP':   {'cpp': 54e-9, 'pitch': 44e-9, 'cap_density': 1.8e-15 / 1e-12, 'res_coeff': 300}
                # source: Intel
            },
            14: {
                'LSTP': {'cpp': 78e-9, 'pitch': 48e-9, 'cap_density': 1.5e-15 / 1e-12, 'res_coeff': 200},
                'HP':   {'cpp': 78e-9, 'pitch': 48e-9, 'cap_density': 1.5e-15 / 1e-12, 'res_coeff': 200}
                # source: AMD
            },
            16: {
                'LSTP': {'cpp': 90e-9, 'pitch': 48e-9, 'cap_density': 1.5e-15 / 1e-12, 'res_coeff': 200},
                'HP':   {'cpp': 90e-9, 'pitch': 48e-9, 'cap_density': 1.5e-15 / 1e-12, 'res_coeff': 200}
                # source: TSMC
            },
            20: {
                'LSTP': {'cpp': 86e-9, 'pitch': 64e-9, 'cap_density': 1.2e-15 / 1e-12, 'res_coeff': 150},
                'HP':   {'cpp': 86e-9, 'pitch': 64e-9, 'cap_density': 1.2e-15 / 1e-12, 'res_coeff': 150}
                # source: IBM 
            }
        }
                
        # standard analog layout overhead
        layout_overhead_factor = 3.0

        try:
            # default to 7nm if not specified
            tech_node = int(params.get('fet_num'))
        except Exception:
            tech_node = 7

        tech_options = node_map.get(tech_node, node_map[7])
        is_hp = bool(params.get('is_hp', False))
        tech = tech_options['HP'] if is_hp else tech_options['LSTP']
        area_m2 = 0.0

        # 1. active device footprint
        for key, val in params.items():
            if not key.startswith('nA'):
                continue
            
            suffix = key[2:]
            width_key = 'nB' + suffix
            
            if width_key in params:
                try:
                    # x-dimension: gate length padded by one cpp
                    l_gate = float(val)
                    x_dim = max(l_gate, tech['cpp']) + tech['cpp']
                    
                    # y-dimension: (fins + 1) * fin pitch
                    num_fins = max(int(params[width_key]), 1)
                    y_dim = (num_fins + 1) * tech['pitch']
                    
                    area_m2 += (x_dim * y_dim)
                except (ValueError, TypeError):
                    continue

        # apply global layout overhead to active area
        area_m2 *= layout_overhead_factor

        # 2. passive devices
        cap_density = tech['cap_density']
        res_area_coeff = tech['res_coeff']
        
        for key, val in params.items():
            try:
                num_val = float(val)
                if num_val <= 0: continue
                
                # nC_x = capacitance (f)
                if key.startswith('nC'):
                    area_m2 += num_val / cap_density
                    
                # nR_x = resistance (ohm)
                elif key.startswith('nR'):
                    area_m2 += num_val * res_area_coeff
            except (ValueError, TypeError):
                continue

        # return as um^2
        return float(area_m2 * 1e12)

    @staticmethod
    def find_vos(results, vcm=0.0):
        """
        Find input-referred offset (vos) in volts.
        """
        dc_offsets, vouts = SpecCalc.extract_dc_sweep(results)
        if dc_offsets is None or vouts is None or len(dc_offsets) < 2:
            return None

        # treat output axis as volts
        vouts = np.asarray(vouts, dtype=float)

        # convert dc offset axis from mv to v
        dc_offsets = np.asarray(dc_offsets, dtype=float) / 1000.0

        # normalize vcm target to volts
        if vcm is not None:
            target = float(vcm)
            if target > 2.0: 
                target = target / 1000.0
        else:
            target = 0.0

        # shift for zero-crossing detection
        y_shifted = vouts - target
        
        # locate sign change
        sign_changes = np.where(np.diff(np.sign(y_shifted)) != 0)[0]

        if len(sign_changes) == 0:
            return None

        # linear interpolation at crossing nearest zero offset
        idx = sign_changes[np.argmin(np.abs(dc_offsets[sign_changes]))]
        
        x1, x2 = dc_offsets[idx], dc_offsets[idx+1]
        y1, y2 = y_shifted[idx], y_shifted[idx+1]

        vos = x1 - y1 * (x2 - x1) / (y2 - y1)

        return float(vos)

    @staticmethod
    def find_output_voltage_swing(results, vcm=0.0, allowed_deviation_pct=10.0):
        """
        Find output swing boundaries in volts using smoothed slope compression.
        S++ Tier: Uses a center-region max gain search to handle systematic offset
        and binary-searches for the contiguous high-gain band.
        """
        dc_offsets, vouts = SpecCalc.extract_dc_sweep(results)
        if dc_offsets is None or vouts is None or len(dc_offsets) < 10:
            return None, None

        try:
            # 1. Standardize units (x: mV -> V, y: V)
            x = np.asarray(dc_offsets, dtype=float) / 1000.0
            y = np.asarray(vouts, dtype=float)

            # 2. Find VOS for region centering
            vos = SpecCalc.find_vos(results, vcm)
            if vos is None:
                return None, None

            # 3. Fit PCHIP and calculate derivative
            # PCHIP is used to ensure the derivative doesn't have artificial ringing
            pchip = interp.PchipInterpolator(x, y, extrapolate=False)
            slope_func = pchip.derivative()

            # 4. Establish baseline Gain (Peak Gain Search)
            # We look for the maximum slope in the central 40% of the sweep
            # to handle designs where peak gain isn't perfectly at VOS.
            x_min, x_max = np.min(x), np.max(x)
            x_range = x_max - x_min
            center_mask = (x > (x_min + 0.3 * x_range)) & (x < (x_max - 0.3 * x_range))
            
            if np.any(center_mask):
                # Sample the center region densely to find the true peak
                x_center = np.linspace(x[center_mask][0], x[center_mask][-1], 500)
                slope_peak = np.max(np.abs(slope_func(x_center)))
            else:
                slope_peak = np.abs(float(slope_func(vos)))

            # 5. Define gain threshold
            # We find the boundaries where gain drops to (1 - deviation) * peak_gain
            threshold = slope_peak * (1.0 - float(allowed_deviation_pct) / 100.0)

            # 6. Dense sampling to locate the contiguous band
            # We use a high-resolution grid to ensure precision
            x_fine = np.linspace(x_min, x_max, 5000)
            slopes = np.abs(slope_func(x_fine))
            in_band = slopes >= threshold

            # 7. Identify the specific contiguous band containing VOS
            vos_idx = np.argmin(np.abs(x_fine - vos))
            if not in_band[vos_idx]:
                # If even VOS is out of band, the circuit is functionally broken
                return None, None

            # Expand left and right until the gain drops below threshold
            left_side = np.where(~in_band[:vos_idx])[0]
            right_side = np.where(~in_band[vos_idx:])[0]

            l_idx = left_side[-1] + 1 if len(left_side) > 0 else 0
            r_idx = (right_side[0] + vos_idx - 1) if len(right_side) > 0 else len(x_fine) - 1

            # 8. Precise Linear Boundary Interpolation
            # This extracts the exact X value where the threshold is crossed
            x_left, x_right = x_fine[l_idx], x_fine[r_idx]

            if l_idx > 0:
                s0, s1 = slopes[l_idx - 1], slopes[l_idx]
                if not np.isclose(s1, s0):
                    x_left = x_fine[l_idx-1] + (threshold - s0) * (x_fine[l_idx] - x_fine[l_idx-1]) / (s1 - s0)

            if r_idx < len(x_fine) - 1:
                s0, s1 = slopes[r_idx], slopes[r_idx + 1]
                if not np.isclose(s1, s0):
                    x_right = x_fine[r_idx] + (threshold - s0) * (x_fine[r_idx+1] - x_fine[r_idx]) / (s1 - s0)

            # 9. Map input boundaries to output voltages
            v_low = float(pchip(x_left))
            v_high = float(pchip(x_right))

            # Return sorted (Min Swing, Max Swing)
            return tuple(sorted((v_low, v_high)))

        except Exception:
            return None, None

    @staticmethod
    def find_psrr(xf_results, gain_ol_lin):
        """
        Calculate Exact Open-Loop PSRR from Unity-Gain XF results.
        Formula: PSRR_ol = A_ol / (A_xf * (1 + A_ol))
        """
        if xf_results is None or gain_ol_lin is None:
            return None
        try:
            # A_ol is the raw linear differential gain (V/V)
            ad = float(np.abs(gain_ol_lin))
            
            # A_xf is the closed-loop leakage measured from supply (V0) to output
            a_cl_leakage = float(np.abs(np.ravel(xf_results.get('V0'))[0]))

            # Handle the "Infinite Rejection" numerical floor
            if a_cl_leakage <= 1e-18:
                return 160.0
            
            # The Exact Ratio: Difference Gain / True Open-Loop Supply Gain
            # A_ol_supply = a_cl_leakage * (1 + ad)
            psrr_linear = ad / (a_cl_leakage * (1 + ad))
            
            psrr_db = float(20 * np.log10(psrr_linear))
            return float(np.clip(psrr_db, -20.0, 160.0))
        except (ZeroDivisionError, ValueError, TypeError):
            return None

    @staticmethod
    def find_cmrr(xf_results, gain_ol_lin):
        """
        Calculate Exact Open-Loop CMRR from Unity-Gain XF results.
        Formula: CMRR_ol = A_ol / (A_xf * (1 + A_ol))
        """
        if xf_results is None or gain_ol_lin is None:
            return None
        try:
            ad = float(np.abs(gain_ol_lin))
            
            # Closed-loop leakage from common-mode source (V1) to output
            a_cl_leakage = float(np.abs(np.ravel(xf_results.get('V1'))[0]))

            if a_cl_leakage <= 1e-18:
                return 160.0
            
            cmrr_linear = ad / (a_cl_leakage * (1 + ad))
            
            cmrr_db = float(20 * np.log10(cmrr_linear))
            return float(np.clip(cmrr_db, -20.0, 160.0))
        except (ZeroDivisionError, ValueError, TypeError):
            return None

    
    @staticmethod
    def find_thd(thd_results):
        """
        Extract total harmonic distortion (THD) in dBc.
        Strictly targets Vout_diff (Differential) with a fallback to Voutp (Single-Ended).
        """
        if thd_results is None:
            return None
        
        sweep_vars = thd_results.get('sweep_vars', [])
        sweep_values = thd_results.get('sweep_values', [])
        is_freq_domain = any(k in sweep_vars for k in ['harmonic', 'freq'])

        # 1. Signal Selection: Strict Two-Case Ladder
        v_sig = None
        if 'Vout_diff' in thd_results:
            # Case 1: Differential Output (Primary)
            v_sig = SpecCalc._coerce_signal_vector(thd_results['Vout_diff'])
        elif 'Voutp' in thd_results:
            # Case 2: Single-Ended Output (Secondary)
            v_sig = SpecCalc._coerce_signal_vector(thd_results['Voutp'])

        if v_sig is None or len(v_sig) == 0:
            return None

        # 2. Spectrum Processing
        if is_freq_domain:
            v_mag = np.abs(v_sig)
            freqs = np.ravel(np.asarray(sweep_values, dtype=float))
            
            # Identify Fundamental (Ignore DC/Low-freq noise below 1Hz)
            pos_mask = freqs > 1.0 
            if not np.any(pos_mask): return None
            
            fund_freq = np.min(freqs[pos_mask])
            fund_idx = np.argmin(np.abs(freqs - fund_freq))
            fund_mag = v_mag[fund_idx]
            
            if fund_mag < 1e-12: return None

            # Industry Standard: Sum the first 9 harmonics (2nd through 10th)
            harm_power = 0.0
            for i in range(2, 11):
                target_f = fund_freq * i
                idx = np.argmin(np.abs(freqs - target_f))
                if np.isclose(freqs[idx], target_f, rtol=0.01):
                    harm_power += v_mag[idx]**2
                    
        else:
            # TRANSIENT PATH (Time-Domain to FFT)
            time = np.asarray(thd_results.get('time', thd_results.get('sweep_values', [])))
            if len(time) < 10:
                return None
            
            # S-Tier Scrubber: Enforce strictly increasing time for FFT integrity
            time, uniq_idx = np.unique(time, return_index=True)
            if len(time) < 10:
                return None
            v_clean = v_sig[uniq_idx]
            
            # Settling Guard: Use only the second half of the simulation
            t_start = max(time) * 0.5
            steady_mask = time >= t_start
            t_steady, v_steady = time[steady_mask], v_clean[steady_mask]
            if len(t_steady) < 8:
                return None
            if t_steady[-1] <= t_steady[0]:
                return None
            
            # Resample to 2^N points for optimal FFT performance
            num_samples = 2**int(np.log2(len(v_steady)))
            if num_samples < 8:
                return None
            t_uniform = np.linspace(t_steady[0], t_steady[-1], num_samples)
            v_uniform = np.interp(t_uniform, t_steady, v_steady)
            
            # Apply Blackman-Harris window for high-dynamic range distortion detection
            window = np.blackman(num_samples)
            v_windowed = v_uniform * window
            
            # FFT with Amplitude Correction for windowing loss
            fft_data = np.fft.rfft(v_windowed)
            fft_mag = np.abs(fft_data) * (2.0 / np.sum(window))
            freqs = np.fft.rfftfreq(num_samples, d=(t_uniform[1] - t_uniform[0]))
            
            # Detect Fundamental (skipping DC at index 0)
            fund_idx = np.argmax(fft_mag[1:]) + 1 
            fund_mag = fft_mag[fund_idx]
            fund_freq = freqs[fund_idx]
            
            if fund_mag < 1e-12: return None
            
            # Harmonic Summation with a 2% frequency window to catch spectral leakage
            harm_power = 0.0
            for i in range(2, 11):
                target_f = fund_freq * i
                h_mask = (freqs >= target_f * 0.98) & (freqs <= target_f * 1.02)
                if np.any(h_mask):
                    harm_power += np.max(fft_mag[h_mask])**2

        # 3. Final THD Calculation
        # Result is in dBc (Decibels relative to the carrier/fundamental)
        total_harm_rms = np.sqrt(max(harm_power, 0.0))
        thd_lin = total_harm_rms / fund_mag
        
        # Floor to -160 dBc to maintain numerical stability in the Bayesian Optimizer
        return float(20 * np.log10(max(thd_lin, 1e-8)))
