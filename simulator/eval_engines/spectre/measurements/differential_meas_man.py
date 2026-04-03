"""
differential_meas_man.py

Author: natelgrw
Last Edited: 04/01/2026

Measurement manager for processing and calculating performance specs
for differential op-amp simulations.
"""

from simulator.eval_engines.spectre.core import EvaluationEngine
import numpy as np
from simulator.eval_engines.spectre.measurements.spec_functions import SpecCalc


# ===== Differential Op-Amp Measurement Manager ===== #


class OpampMeasMan(EvaluationEngine):
    """
    Measurement manager for differential op-amp simulations.
    Mirrors single_ended_meas_man.py structure and logic.

    Initialization Parameters:
    --------------------------
    config (dict): Configuration dictionary for the measurement manager.
    """

    def __init__(self, config):

        EvaluationEngine.__init__(self, config)

    def process_ac(self, results, params):
        """
        Calls SpecCalc functions to extract specs from AC analysis results, 
        handling differential output result formats.

        Parameters:
        -----------
        results (dict): Dictionary containing simulation results from various analyses.
        params (dict): Dictionary containing simulation parameters.

        Returns:
        --------
        dict: Dictionary of extracted performance specifications.
        """
        return ACTB.process_ac(results, params)

    def get_specs(self, results_dict, params):
        """
        Extracts performance specifications from the results dictionary, handling
        differential result formats.

        Parameters:
        -----------
        results_dict (dict): Dictionary containing simulation results from various analyses.
        params (dict): Dictionary containing simulation parameters.

        Returns:
        --------
        dict: Dictionary of extracted performance specifications.
        """
        if results_dict and isinstance(results_dict, dict):
            keys = list(results_dict.keys())
            if len(keys) == 1 and isinstance(results_dict[keys[0]], tuple):
                results_dict = results_dict[keys[0]][1]
        if 'gain_ol_dc_db' in results_dict or 'ugbw_hz' in results_dict:
            return results_dict
        if 'ac_dc' in results_dict:
            ac_dc_tuple = results_dict['ac_dc']
            return ac_dc_tuple[1]
        else:
            return self.process_ac(results_dict, params)


# ===== AC Analysis Function Class for OpampMeasMan ===== #


class ACTB:
    """
    AC Analysis Trait Base for differential OpampMeasMan.
    Mirrors single ended extraction logic with differential extraction.
    """

    @classmethod
    def process_ac(cls, results, params):
        """
        Process AC analysis results to extract performance specifications for a differential op-amp.

        Parameters:
        -----------
        results (dict): Dictionary containing simulation results from various analyses.
        params (dict): Dictionary containing simulation parameters.

        Returns:
        --------
        dict: Dictionary of extracted performance specifications.
        """
        if not isinstance(results, dict):
            return {}

        # individual simulation results
        ac_result_se    = results.get('stb_sim')
        dc_results      = results.get('dcOp_sim')
        noise_results   = results.get('noise_sim')
        xf_resultsdict  = results.get('xf_sim')
        thd_results     = results.get('thd_pss.fd')
        slew_results    = results.get('slew_large_tran')
        settle_results  = results.get('settle_small_tran')

        # initializing all spec variables
        vos_v = gain_ol_dc_db = gain_ol_lin = ugbw_hz = pm_deg = None
        estimated_area_um2 = power_w = cmrr_dc_db = psrr_dc_db = None
        integrated_noise_vrms = slew_rate_v_us = settle_time_ns = thd_db = None
        valid = False
        stability_valid = False

        # initialization of dcOp specs for each transistor
        mm_names = []
        ids_MM = {}
        gm_MM = {}
        gds_MM = {}
        vth_MM = {}
        vdsat_MM = {}
        vgs_MM = {}
        vds_MM = {}
        cgg_MM = {}
        cgs_MM = {}
        cdd_MM = {}
        cgd_MM = {}
        css_MM = {}
        region_MM = {}

        # dc sweep processing
        swing_low_v = swing_high_v = output_voltage_swing_range_v = None
        vos_v = None
        vdd_val = 0.0
        if dc_results:
            mm_names = sorted(list(set(comp.split(':')[0] for comp in dc_results.keys() if comp.startswith('MM'))))
            ids_MM    = {mm: np.nan for mm in mm_names}
            gm_MM     = {mm: np.nan for mm in mm_names}
            gds_MM    = {mm: np.nan for mm in mm_names}
            vth_MM    = {mm: np.nan for mm in mm_names}
            vdsat_MM  = {mm: np.nan for mm in mm_names}
            vgs_MM    = {mm: np.nan for mm in mm_names}
            vds_MM    = {mm: np.nan for mm in mm_names}
            cgg_MM    = {mm: np.nan for mm in mm_names}
            cgs_MM    = {mm: np.nan for mm in mm_names}
            cdd_MM    = {mm: np.nan for mm in mm_names}
            cgd_MM    = {mm: np.nan for mm in mm_names}
            css_MM    = {mm: np.nan for mm in mm_names}
            region_MM = {mm: np.nan for mm in mm_names}

            if params and 'vdd' in params:
                vdd_val = float(params['vdd'])
            elif results.get('vdd'):
                vdd_val = float(results['vdd'])

            if 'V0:p' in dc_results:
                i_supply = np.abs(dc_results['V0:p'])
                power_w = i_supply * vdd_val if vdd_val > 0 else i_supply

            for comp, val in dc_results.items():
                if not comp.startswith('MM'):
                    continue
                base = comp.split(':')[0]
                suffix = comp.split(':')[1] if ':' in comp else ''
                try:
                    fval = float(np.abs(val))
                except Exception:
                    fval = val
                if suffix == 'ids': ids_MM[base] = fval
                elif suffix == 'gm': gm_MM[base] = fval
                elif suffix == 'gds': gds_MM[base] = fval
                elif suffix == 'vth': vth_MM[base] = fval
                elif suffix == 'vdsat': vdsat_MM[base] = fval
                elif suffix == 'vgs': vgs_MM[base] = fval
                elif suffix == 'vds': vds_MM[base] = fval
                elif suffix == 'cgg': cgg_MM[base] = fval
                elif suffix == 'cgs': cgs_MM[base] = fval
                elif suffix == 'cdd': cdd_MM[base] = fval
                elif suffix == 'cgd': cgd_MM[base] = fval
                elif suffix == 'css': css_MM[base] = fval
                elif suffix == 'region': region_MM[base] = fval

            vos_v = SpecCalc.find_vos(results)
            swing_low_v, swing_high_v = SpecCalc.find_output_voltage_swing(results)
            output_voltage_swing_range_v = swing_high_v - swing_low_v if (swing_high_v is not None and swing_low_v is not None) else None
        else:
            swing_low_v = swing_high_v = output_voltage_swing_range_v = None

        # stability analysis processing (strict STB: loopGain only)
        if ac_result_se:
            loop_gain = ac_result_se.get('loopGain')
            if loop_gain is not None:
                vout = np.array(loop_gain)
            else:
                vout = np.array([])

            freq = ac_result_se.get('sweep_values')
            if freq is None:
                keys = list(ac_result_se.keys())
                freq_key = next((k for k in keys if 'freq' in k.lower()), None)
                if freq_key:
                    freq = ac_result_se[freq_key]

            if len(vout) > 0 and freq is not None and len(freq) > 0:
                min_ac_len = min(len(freq), len(vout))
                freq_clean = freq[:min_ac_len]
                vout_clean = vout[:min_ac_len]

                gain_ol_lin = SpecCalc.find_dc_gain(vout_clean)
                ugbw_hz, stability_valid = SpecCalc.find_ugbw(freq_clean, vout_clean)
                pm_deg = SpecCalc.find_phm(freq_clean, vout_clean, ugbw_hz, stability_valid)
            elif len(vout) > 0:
                gain_ol_lin = SpecCalc.find_dc_gain(vout)

        if params:
            estimated_area_um2 = SpecCalc.find_estimated_area(params)

        if xf_resultsdict and gain_ol_lin is not None:
            cmrr_dc_db = SpecCalc.find_cmrr(xf_resultsdict)

        if xf_resultsdict and gain_ol_lin is not None:
            psrr_dc_db = SpecCalc.find_psrr(xf_resultsdict)

        if gain_ol_lin is not None and gain_ol_lin != 0:
            gain_ol_dc_db = 20 * np.log10(np.abs(gain_ol_lin))

        if noise_results:
            integrated_noise_vrms = SpecCalc.find_integrated_noise(noise_results)

        if slew_results:
            time = slew_results.get('time', [])
            t_val_p = slew_results.get('Voutp')
            t_val_n = slew_results.get('Voutn')
            if (time is None or len(time) == 0) and 'sweep_values' in slew_results:
                time = slew_results['sweep_values']

            if t_val_p is not None and len(t_val_p) > 0:
                if t_val_n is not None and len(t_val_n) > 0:
                    min_t_len = min(len(t_val_p), len(t_val_n))
                    diff_tran = np.array(t_val_p[:min_t_len]) - np.array(t_val_n[:min_t_len])
                else:
                    diff_tran = None

                if diff_tran is not None:
                    min_len = min(len(time), len(diff_tran))
                    slew_rate_v_us = SpecCalc.find_slew_rate(list(zip(time[:min_len], diff_tran[:min_len])))

        if settle_results:
            time = settle_results.get('time', [])
            t_val_p = settle_results.get('Voutp')
            t_val_n = settle_results.get('Voutn')
            if (time is None or len(time) == 0) and 'sweep_values' in settle_results:
                time = settle_results['sweep_values']

            if t_val_p is not None and len(t_val_p) > 0:
                if t_val_n is not None and len(t_val_n) > 0:
                    min_t_len = min(len(t_val_p), len(t_val_n))
                    diff_tran = np.array(t_val_p[:min_t_len]) - np.array(t_val_n[:min_t_len])
                else:
                    diff_tran = None

                if diff_tran is not None:
                    min_len = min(len(time), len(diff_tran))
                    settle_time_ns = SpecCalc.find_settle_time(list(zip(time[:min_len], diff_tran[:min_len])), vdd_val)

        if thd_results:
            thd_db = SpecCalc.find_thd(thd_results)

        # operating-point health: all transistors should be in region 2
        ops_good = False
        if region_MM:
            ops_good = True
            for mm in mm_names:
                reg_val = region_MM.get(mm, np.nan)
                if not np.isfinite(reg_val) or not np.isclose(float(reg_val), 2.0, atol=0.25):
                    ops_good = False
                    break

        gain_ol_dc_db_valid = gain_ol_dc_db is not None
        ugbw_hz_valid = (ugbw_hz is not None) and bool(stability_valid)
        pm_deg_valid = (pm_deg is not None) and bool(stability_valid)
        estimated_area_um2_valid = estimated_area_um2 is not None
        power_w_valid = power_w is not None
        vos_v_valid = vos_v is not None
        cmrr_dc_db_valid = cmrr_dc_db is not None
        psrr_dc_db_valid = psrr_dc_db is not None
        thd_db_valid = thd_db is not None
        output_voltage_swing_range_v_valid = (
            swing_low_v is not None and swing_high_v is not None and output_voltage_swing_range_v is not None and output_voltage_swing_range_v > 0.0
        )
        integrated_noise_vrms_valid = integrated_noise_vrms is not None
        slew_rate_v_us_valid = (slew_rate_v_us is not None) and (slew_rate_v_us > 0.0)
        settle_time_ns_valid = (settle_time_ns is not None) and (settle_time_ns > 0.0)

        return dict(
            gain_ol_dc_db = gain_ol_dc_db if gain_ol_dc_db is not None else np.nan,
            ugbw_hz = ugbw_hz if ugbw_hz is not None else np.nan,
            pm_deg = pm_deg if pm_deg is not None else np.nan,
            estimated_area_um2 = estimated_area_um2 if estimated_area_um2 is not None else np.nan,
            power_w = power_w if power_w is not None else np.nan,
            vos_v = vos_v if vos_v is not None else np.nan,
            cmrr_dc_db = cmrr_dc_db if cmrr_dc_db is not None else np.nan,
            psrr_dc_db = psrr_dc_db if psrr_dc_db is not None else np.nan,
            thd_db = thd_db if thd_db is not None else np.nan,
            output_voltage_swing_range_v = output_voltage_swing_range_v if output_voltage_swing_range_v is not None else np.nan,
            output_voltage_swing_min_v = swing_low_v if swing_low_v is not None else np.nan,
            output_voltage_swing_max_v = swing_high_v if swing_high_v is not None else np.nan,
            integrated_noise_vrms = integrated_noise_vrms if integrated_noise_vrms is not None else np.nan,
            slew_rate_v_us = slew_rate_v_us if (slew_rate_v_us is not None and slew_rate_v_us > 0.0) else np.nan,
            settle_time_ns = settle_time_ns if (settle_time_ns is not None and settle_time_ns > 0.0) else np.nan,
            gain_ol_dc_db_valid = gain_ol_dc_db_valid,
            ugbw_hz_valid = ugbw_hz_valid,
            pm_deg_valid = pm_deg_valid,
            estimated_area_um2_valid = estimated_area_um2_valid,
            power_w_valid = power_w_valid,
            vos_v_valid = vos_v_valid,
            cmrr_dc_db_valid = cmrr_dc_db_valid,
            psrr_dc_db_valid = psrr_dc_db_valid,
            thd_db_valid = thd_db_valid,
            output_voltage_swing_range_v_valid = output_voltage_swing_range_v_valid,
            integrated_noise_vrms_valid = integrated_noise_vrms_valid,
            slew_rate_v_us_valid = slew_rate_v_us_valid,
            settle_time_ns_valid = settle_time_ns_valid,
            zregion_of_operation_MM = region_MM,
            zzids_MM = ids_MM,
            zzvds_MM = vds_MM,
            zzvgs_MM = vgs_MM,
            zzgm_MM = gm_MM,
            zzgds_MM = gds_MM,
            zzvth_MM = vth_MM,
            zzvdsat_MM = vdsat_MM,
            zzcgg_MM = cgg_MM,
            zzcgs_MM = cgs_MM,
            zzcdd_MM = cdd_MM,
            zzcgd_MM = cgd_MM,
            zzcss_MM = css_MM,
        )
