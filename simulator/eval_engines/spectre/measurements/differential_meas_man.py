"""
differential_meas_man.py

Author: natelgrw
Last Edited: 01/08/2026

Measurement manager for processing and calculating 11 performance specs 
for differential op-amp simulations.
"""

from simulator.eval_engines.spectre.core import EvaluationEngine
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt
import scipy.integrate as scint
from simulator import globalsy


# ===== Differential Op-Amp Measurement Manager ===== #


class OpampMeasMan(EvaluationEngine):
    """
    Measurement manager for differential op-amp simulations.

    Supports the calculation of 11 performance specs:
    - Gain
    - UGBW
    - Phase Margin
    - Power Consumption
    - CMRR
    - Input Offset Voltage (Vos)
    - Linearity Range
    - Output Voltage Swing
    - Integrated Noise
    - Slew Rate
    - Settling Time

    Built on top of the EvaluationEngine base class.
    """

    def __init__(self, yaml_fname):

        EvaluationEngine.__init__(self, yaml_fname)

    def get_specs(self, results_dict):
        """
        Constructs a cleaned specs dictionary from an input results dictionary.

        Parameters:
        -----------
        results_dict: dict
            The raw results dictionary from simulations.
        
        Returns:
        --------
        specs_dict: dict
            The cleaned specs dictionary extracted from results.
        """
        specs_dict = dict()
        ac_dc = results_dict['ac_dc']

        # extracting specs from ac_dc results
        for _, res, _ in ac_dc:
            specs_dict = res

        return specs_dict

    def compute_penalty(self, spec_nums, spec_kwrd):
        """
        Computes penalties for given spec numbers based on predefined spec ranges.

        Parameters:
        -----------
        spec_nums: list
            The spec numbers to evaluate.
        spec_kwrd: str
            The keyword identifying the spec in the spec_range dictionary.  

        Returns:
        --------
        penalties: list
            List of computed penalties for each spec number.
        """
        if type(spec_nums) is not list:
            spec_nums = [spec_nums]

        penalties = []

        # compute penalties for each spec number
        for spec_num in spec_nums:
            penalty = 0
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

    Provides methods to process AC analysis results and compute 
    11 performance specs from simulation data.
    """

    @classmethod
    def process_ac(self, results, params):
        """
        Processes AC analysis results to compute performance specifications.

        Parameters:
        -----------
        results: dict
            The raw results dictionary derived from simulations.
        params: dict
            Additional parameters for processing.
        
        Returns:
        --------
        specs: dict 
            A cleaned specs dictionary with computed performance specifications.
        """
        # AC, DC, noise, and transient raw extraction
        ac_result_diff = results['acswp-000_ac']
        ac_result_cm = results['acswp-001_ac']

        dc_results = results['dcswp-500_dcOp']

        noise_results = results["noise"]

        tran_results_p = results["tran_voutp"]
        tran_results_n = results["tran_voutn"]
        tran_results = self.combine_tran(tran_results_p, tran_results_n)

        # common mode voltage 
        vcm = dc_results["cm"]

        # differential output voltage
        vout_diff_p = ac_result_diff['Voutp']
        vout_diff_n = ac_result_diff['Voutn']
        vout_diff = vout_diff_p - vout_diff_n

        # frequency vector
        freq = ac_result_diff['sweep_values']

        # Extract MM transistor metrics
        ids_MM = {}
        gm_MM = {}
        vgs_MM = {}
        vds_MM = {}
        region_MM = {}

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

        # compute performance specs
        vos = self.find_vos(results, vcm)
        gain = self.find_dc_gain(vout_diff)
        ugbw, valid = self.find_ugbw(freq, vout_diff)
        phm = self.find_phm(freq, vout_diff)
        power = -dc_results['V0:p']
        cmrr = self.find_cmrr(vout_diff, ac_result_cm['Voutp'])
        linearity = self.find_linearity(results, vout_diff)
        output_voltage_swing = self.find_output_voltage_swing(results, vcm)
        integrated_noise = self.find_integrated_noise(noise_results)
        slew_rate = self.find_slew_rate(tran_results)
        settle_time = self.find_settle_time(tran_results)


        results = dict(
            gain = gain,
            ugbw = ugbw,
            pm = phm,
            power = power,
            vos = vos,
            cmrr = cmrr,
            linearity = linearity,
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
    def find_dc_gain(self, vout_diff):
        """
        Finds the DC gain from differential output voltage.

        Parameters:
        -----------
        vout_diff: numpy array
            The differential output voltage array from AC analysis.
        
        Returns:
        --------
        dc_gain: float
            The DC gain value.
        """
        return np.abs(vout_diff)[0]

    @classmethod
    def find_ugbw(self, freq, vout_diff):
        """
        Finds unity gain bandwidth (UGBW) from differential output voltage.

        Parameters:
        -----------
        freq: numpy array
            The frequency array from AC analysis.
        vout_diff: numpy array
            The differential output voltage array from AC analysis.
        
        Returns:
        --------
        ugbw: float
            The unity gain bandwidth value.
        valid: bool
            Indicates if a valid UGBW crossing was found.
        """
        gain = np.abs(vout_diff)
        ugbw, valid = self._get_best_crossing(freq, gain, val=1)
        if valid:
            return ugbw, valid
        else:
            return freq[0], valid

    @classmethod
    def find_phm(self, freq, vout_diff):
        """
        Finds phase margin (PM) from differential output voltage.

        Parameters:
        -----------
        freq: numpy array
            The frequency array from AC analysis.
        vout_diff: numpy array
            The differential output voltage array from AC analysis.
        
        Returns:
        --------
        pm: float
            The phase margin value. If UGBW is not found, returns -180.
        """
        gain = np.abs(vout_diff)

        # calculate phase in degrees and unwrap
        phase = np.angle(vout_diff, deg=True)
        phase = np.unwrap(np.deg2rad(phase))
        phase = np.rad2deg(phase)

        # create interpolation function for phase and find phase at UGBW
        phase_fun = interp.interp1d(freq, phase, kind='quadratic')
        ugbw, valid = self._get_best_crossing(freq, gain, val=1)

        # wrap phase into [-180, 180]
        if valid:
            phase_at_ugbw = (phase_fun(ugbw) + 180) % 360 - 180
            pm = 180 + phase_at_ugbw
            pm = (pm + 180) % 360 - 180
            return pm
        else:
            return -180

    @classmethod
    def find_cmrr(self, vout_diff, vout_cm):
        """
        Finds common mode rejection ratio (CMRR) from 
        differential and common mode output voltages.

        Parameters:
        -----------
        vout_diff: numpy array
            The differential output voltage array.
        vout_cm: numpy array
            The common mode output voltage array.
        
        Returns:
        --------
        cmrr: float
            The common mode rejection ratio value.  
        """
        gain_diff = self.find_dc_gain(vout_diff)
        gain_cm = np.abs(vout_cm)[0]

        return gain_diff / gain_cm

    @classmethod
    def extract_dc_sweep(self, results):
        """
        Extracts and returns sorted (dc_offsets, vouts) arrays from a DC sweep.

        Parameters:
        -----------
        results: dict
            The raw results dictionary from simulations.
        
        Returns:
        --------
        dc_offsets: numpy array
            Sorted array of DC offset values.
        vouts: numpy array
            Sorted array of output voltages corresponding to DC offsets.
        """
        dc_offsets = []
        vouts = []

        for result in results.keys():
            if result.startswith('dcswp-'):
                val = results[result]["Voutp"]
                # parse offset value from key string
                if result[6:10] == "1000":
                    dc_offset = int(result[6:10]) * 0.001 - 0.5
                else:
                    dc_offset = int(result[6:9]) * 0.001 - 0.5
                dc_offsets.append(dc_offset)
                vouts.append(val)

        # sort both arrays by offset
        dc_offsets = np.array(dc_offsets)
        vouts = np.array(vouts)
        sort_idx = np.argsort(dc_offsets)

        return dc_offsets[sort_idx], vouts[sort_idx]

    @classmethod
    def find_vos(self, results, vcm):
        """
        Finds input offset voltage (Vos) from DC sweep results.

        Parameters:
        -----------
        results: dict
            The raw results dictionary from simulations.
        vcm: float
            The common mode voltage.    
        
        Returns:
        --------
        vos: float
            The input offset voltage value. 
        """
        dc_offsets, vouts = self.extract_dc_sweep(results)

        # create a smooth cubic spline across all points
        spline = interp.UnivariateSpline(dc_offsets, vouts, s=0)

        def root_func(x):
            """
            Calculates difference between spline output and target vcm.

            Parameters:
            -----------
            x: float
                The DC offset value.

            Returns:
            --------
            diff: float
                The difference between spline output and vcm.
            """
            return spline(x) - vcm

        # find Vos using Brent's method
        try:
            vos = sciopt.brentq(root_func, dc_offsets[0], dc_offsets[-1])
        except ValueError:
            vos = -1e2

        return vos

    @classmethod
    def find_linearity(self, results, vout_diff, allowed_deviation_pct=2.0):
        """
        Finds gain linearity range from DC sweep results.

        Parameters:
        -----------
        results: dict
            The raw results dictionary from simulations.
        vout_diff: numpy array
            The differential output voltage array.
        allowed_deviation_pct: float
            The allowed percentage deviation from ideal linearity.
        
        Returns:
        --------
        linear_range: tuple or None
            The (min, max) DC offset values defining the linearity range.
            Returns None if gain is too small to define linearity.  
        """
        # find gain
        gain = self.find_dc_gain(vout_diff)
        if gain < 1:
            return None

        dc_offsets, vouts = self.extract_dc_sweep(results)

        # spline operations
        spline = interp.UnivariateSpline(dc_offsets, vouts, s=0)

        slope_spline = spline.derivative(n=1)

        fine_x = np.linspace(dc_offsets.min(), dc_offsets.max(), 2000)
        fine_slope = slope_spline(fine_x)

        zero_idx = np.argmin(np.abs(fine_x - 0))
        slope_at_zero = fine_slope[zero_idx]

        # integrating allowed deviation in slope
        allowed_dev = abs(slope_at_zero) * (allowed_deviation_pct / 100.0)

        # expand left
        left_idx = zero_idx
        while left_idx > 0 and abs(fine_slope[left_idx] - slope_at_zero) <= allowed_dev:
            left_idx -= 1

        # expand right
        right_idx = zero_idx
        while right_idx < len(fine_x) - 1 and abs(fine_slope[right_idx] - slope_at_zero) <= allowed_dev:
            right_idx += 1

        linear_range = (fine_x[left_idx], fine_x[right_idx])
        return linear_range

    @classmethod
    def find_output_voltage_swing(self, results, vcm, allowed_deviation_pct=10.0):
        """
        Finds output voltage swing from DC sweep results.

        Parameters:
        -----------
        results: dict
            The raw results dictionary from simulations.
        vcm: float
            The common mode voltage.
        allowed_deviation_pct: float
            The allowed percentage deviation from max slope.
        
        Returns:
        --------
        y_range: tuple
            The (y_min, y_max) output voltage swing range.
        """
        # extract dc sweep data
        dc_offsets, vouts = self.extract_dc_sweep(results)
        dc_offsets = np.array(dc_offsets)
        vouts = np.array(vouts)

        # spline operations
        spline = interp.UnivariateSpline(dc_offsets, vouts, s=0)

        slope_spline = spline.derivative()

        # Vos operations
        idx_vos = np.argmin(np.abs(vouts - vcm))
        vos = dc_offsets[idx_vos]

        max_slope = slope_spline(vos)

        # allowed slope deviation
        allowed_dev = abs(max_slope) * (allowed_deviation_pct / 100)

        # expand left
        idx_left = idx_vos
        while idx_left > 0:
            if abs(slope_spline(dc_offsets[idx_left]) - max_slope) > allowed_dev:
                break
            idx_left -= 1

        # expand right
        idx_right = idx_vos
        while idx_right < len(dc_offsets) - 1:
            if abs(slope_spline(dc_offsets[idx_right]) - max_slope) > allowed_dev:
                break
            idx_right += 1

        # Get Y range
        y_min = vouts[idx_left]
        y_max = vouts[idx_right]

        return y_min, y_max
    
    @classmethod
    def find_integrated_noise(self, noise_results):
        """
        Integrates total noise PSD over frequency for all 
        transistors starting with 'MM' and sum.

        Parameters:
        -----------
        noise_results: dict
            The noise results dictionary from simulations.

        Returns:
        --------
        total_integrated_noise: float
            Total integrated noise power (V^2) summed across all transistors.
        """
        total_integrated_noise = 0.0

        # fixed number of frequency points per transistor
        num_points = 55

        # frequency range: 1 MHz to 500 MHz
        f_start = 1e6
        f_stop = 5e8
        freqs = np.logspace(np.log10(f_start), np.log10(f_stop), num_points)

        for key, noise_array in noise_results.items():
            if not key.startswith("MM"):
                continue

            if len(noise_array) != num_points:
                raise ValueError(f"Expected 55 points for {key}, but got {len(noise_array)}")

            total_psd = np.array([entry[b'total'] for entry in noise_array])

            # integrate noise PSD vs frequency to get noise power (V^2)
            integrated_noise = scint.simps(total_psd, freqs)

            total_integrated_noise += integrated_noise

        return total_integrated_noise
    
    @classmethod
    def combine_tran(self, tranp_results, trann_results):
        """
        Combines positive and negative transient results into differential output.

        Parameters:
        -----------
        tranp_results: list of tuples
            Transient results for positive output (time, Voutp).
        trann_results: list of tuples
            Transient results for negative output (time, Voutn).
        
        Returns:
        --------
        combined_results: list of tuples
            Combined transient results (time, Voutp - Voutn).
        """
        time = np.array([t for t, _ in tranp_results])
        voutp = np.array([v for _, v in tranp_results])
        voutn = np.array([v for _, v in trann_results])

        return list(zip(time, voutp - voutn))

    @classmethod
    def find_slew_rate(self, tran_results):
        """
        Finds slew rate from transient results.

        Parameters:
        -----------
        tran_results: list of tuples
            Transient results (time, Vout).
        
        Returns:
        --------
        slew_rate: float
            The slew rate value (mV/ns -> V/μs).
        """
        time = np.array([t for t, _ in tran_results])
        vout = np.array([v for _, v in tran_results])

        # spline interpolation of Vout vs time
        spline = interp.CubicSpline(time, vout)
        time_fine = np.linspace(time[0], time[-1], 50000)

        # calculate derivative at fine points
        dv_dt = spline.derivative()(time_fine)

        # max absolute slope = slew rate
        slew_rate = np.max(np.abs(dv_dt))

        return slew_rate

    @classmethod
    def find_settle_time(self, tran_results, tol=0.005,
                        delay=1e-9, change=50e-12, width=100e-9):
        """
        Finds settling time from transient results.

        Parameters:
        -----------
        tran_results: list of tuples
            Transient results (time, Vout).
        tol: float
            The tolerance band percentage for settling.
        delay: float
            The input delay time in seconds.
        change: float
            The input change time in seconds.
        width: float
            The input pulse width in seconds.       

        Returns:
        --------
        settle_time: float or None
            The settling time value (s). Returns None if never settles.
        """

        time = np.array([t for t, _ in tran_results])
        vout = np.array([v for _, v in tran_results])

        # spline interpolation of Vout vs time
        spline = interp.CubicSpline(time, vout)
        time_fine = np.linspace(time[0], time[-1], 50000)
        vout_fine = spline(time_fine)

        # average of last 5% of samples
        n_tail = max(5, len(vout_fine) // 20)
        v_final = float(np.mean(vout_fine[-n_tail:]))

        lower, upper = v_final * (1 - tol), v_final * (1 + tol)

        # Check where Vout enters the ±tol band
        in_band = (vout_fine >= lower) & (vout_fine <= upper)
        stays_in = np.flip(np.cumprod(np.flip(in_band).astype(int)).astype(bool))
        idx = np.argmax(stays_in)

        if not stays_in[idx]:
            return None 

        t_out = time_fine[idx]

        # fixed input 50% reference time
        t_ref = (delay + width + change/2) * 1e9

        return max(0.0, t_out - t_ref)

    @classmethod
    def _get_best_crossing(self, xvec, yvec, val):
        """
        Finds the best crossing point where yvec crosses val using interpolation.

        Parameters:
        -----------
        xvec: numpy array
            The x values array.
        yvec: numpy array
            The y values array.
        val: float
            The target value to find crossing for.
        
        Returns:
        --------
        crossing_x: float
            The x value where yvec crosses val.
        valid: bool
            Indicates if a valid crossing was found.
        """
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]

        try:
            return sciopt.brentq(fzero, xstart, xstop), True
        except ValueError:
            return xstop, False
