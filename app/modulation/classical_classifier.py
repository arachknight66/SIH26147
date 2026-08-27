from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .models import ModulationFamily, ModulationFeatureVector

@dataclass
class ClassicalScores:
    scores: dict[tuple[ModulationFamily, int], float]
    evidence_breakdown: dict[tuple[ModulationFamily, int], dict[str, float]]
    supporting_notes: dict[tuple[ModulationFamily, int], list[str]]
    contradictions: dict[tuple[ModulationFamily, int], list[str]]

def compute_classical_scores(fv: ModulationFeatureVector) -> ClassicalScores:
    """
    Compute deterministic classical modulation scores and physical evidence vectors.

    Parameters
    ----------
    fv : ModulationFeatureVector

    Returns
    -------
    ClassicalScores
    """
    scores: dict[tuple[ModulationFamily, int], float] = {}
    evidence: dict[tuple[ModulationFamily, int], dict[str, float]] = {}
    notes: dict[tuple[ModulationFamily, int], list[str]] = {}
    contra: dict[tuple[ModulationFamily, int], list[str]] = {}

    amp = fv.amplitude
    ph = fv.phase
    fr = fv.frequency
    cum = fv.cumulants
    spec = fv.spectral
    cyc = fv.cyclostationary

    has_fsk_states = (fr.bimodal_separation is not None and 0.03 <= fr.bimodal_separation <= 0.45 and fr.bimodal_prominence >= 0.20)
    has_digital_clock = (cyc.periodicity_score >= 0.25)

    # =========================================================================
    # 1. 2-FSK / BFSK Evaluation
    # =========================================================================
    fsk_amp_score = float(np.clip(1.0 - (amp.norm_variance / 0.20), 0.05, 0.99))
    fsk_notes = []
    fsk_contra = []

    if has_fsk_states:
        fsk_freq_score = float(np.clip(0.60 + 0.30 * fr.bimodal_prominence + 0.10 * fr.state_occupancy_ratio, 0.40, 0.99))
        fsk_notes.append(f"Instantaneous frequency exhibits bimodal states (delta_f = {fr.bimodal_separation:.4f} cycles/sample).")
    else:
        fsk_freq_score = 0.05
        fsk_contra.append("Instantaneous frequency histogram does not exhibit 2 distinct frequency states.")

    if cum.f20 > 0.40:
        fsk_contra.append(f"High non-circular cumulant f20={cum.f20:.2f} contradicts circular FSK.")

    fsk_cum_score = float(np.clip(1.0 - cum.f20 - abs(cum.f42 - 1.0), 0.05, 0.95))
    fsk_total = 0.20 * fsk_amp_score + 0.50 * fsk_freq_score + 0.15 * fsk_cum_score + 0.15 * cyc.periodicity_score
    if not has_fsk_states:
        fsk_total *= 0.20

    scores[(ModulationFamily.FSK, 2)] = round(float(np.clip(fsk_total, 0.0, 0.99)), 3)
    evidence[(ModulationFamily.FSK, 2)] = {
        "amplitude": fsk_amp_score,
        "phase": 0.5,
        "frequency": fsk_freq_score,
        "cumulants": fsk_cum_score,
        "spectral": 0.5,
        "periodicity": cyc.periodicity_score,
    }
    notes[(ModulationFamily.FSK, 2)] = fsk_notes
    contra[(ModulationFamily.FSK, 2)] = fsk_contra

    # =========================================================================
    # 2. BPSK Evaluation
    # =========================================================================
    bpsk_amp_score = float(np.clip(1.0 - (amp.norm_variance / 0.18), 0.05, 0.99))
    bpsk_cum_score = float(np.clip(cum.f20 * 0.70 + min(cum.f40 / 1.5, 1.0) * 0.30, 0.05, 0.99))
    bpsk_ph_score = float(np.clip(1.0 - ph.var_phase_sq, 0.05, 0.99))
    bpsk_notes = []
    bpsk_contra = []

    if has_fsk_states:
        bpsk_contra.append("Bimodal instantaneous frequency indicates FSK rather than single-carrier BPSK.")

    if not has_digital_clock and amp.norm_variance > 0.05:
        bpsk_contra.append("Lack of digital symbol clock periodicity contradicts digital BPSK.")

    if amp.norm_variance >= 0.18:
        bpsk_contra.append(f"High envelope variance ({amp.norm_variance:.3f}) indicates AM or fading rather than BPSK.")

    # AM carrier rejection: AM is all-positive real (mean > 0.6*RMS), whereas BPSK is zero-mean antipodal (+-1)
    if amp.mean > 0.60 and amp.norm_variance > 0.10:
        bpsk_contra.append("High DC envelope offset indicates AM with carrier rather than antipodal BPSK.")

    if cum.f20 >= 0.50:
        bpsk_notes.append(f"Strong non-zero 2nd-order unconjugated cumulant f20={cum.f20:.3f} (theoretical BPSK f20=1.0).")
    else:
        bpsk_contra.append(f"Low f20={cum.f20:.3f} is inconsistent with real 1D BPSK.")

    if ph.var_phase_sq <= 0.35:
        bpsk_notes.append(f"Squaring phase variance is low ({ph.var_phase_sq:.3f}), indicating 180-deg phase state collapse.")

    bpsk_total = 0.15 * bpsk_amp_score + 0.40 * bpsk_cum_score + 0.30 * bpsk_ph_score + 0.15 * cyc.periodicity_score
    if cum.f20 < 0.30 or has_fsk_states or amp.norm_variance >= 0.18:
        bpsk_total *= 0.20
    if not has_digital_clock and amp.norm_variance > 0.05:
        bpsk_total *= 0.20

    scores[(ModulationFamily.PSK, 2)] = round(float(np.clip(bpsk_total, 0.0, 0.99)), 3)
    evidence[(ModulationFamily.PSK, 2)] = {
        "amplitude": bpsk_amp_score,
        "phase": bpsk_ph_score,
        "frequency": 0.5,
        "cumulants": bpsk_cum_score,
        "spectral": 0.5,
        "periodicity": cyc.periodicity_score,
    }
    notes[(ModulationFamily.PSK, 2)] = bpsk_notes
    contra[(ModulationFamily.PSK, 2)] = bpsk_contra

    # =========================================================================
    # 3. QPSK Evaluation
    # =========================================================================
    qpsk_amp_score = float(np.clip(1.0 - (amp.norm_variance / 0.25), 0.05, 0.99))
    # Circular C20 approx 0, C40 approx 0.8-1.0, C42 approx 0.8-1.0
    qpsk_cum_score = float(np.clip((1.0 - min(cum.f20 / 0.40, 1.0)) * 0.4 + (1.0 - min(abs(cum.f40 - 0.85) / 0.7, 1.0)) * 0.3 + (1.0 - min(abs(cum.f42 - 0.85) / 0.7, 1.0)) * 0.3, 0.05, 0.99))
    qpsk_ph_score = float(np.clip((1.0 - min(ph.var_phase_4th / 0.70, 1.0)) * 0.65 + min(ph.var_phase_sq / 0.6, 1.0) * 0.35, 0.05, 0.99))
    qpsk_notes = []
    qpsk_contra = []

    if has_fsk_states:
        qpsk_contra.append("Bimodal instantaneous frequency indicates FSK rather than QPSK.")

    if not has_digital_clock:
        qpsk_contra.append("Lack of digital symbol clock periodicity contradicts digital QPSK.")

    if cum.f20 <= 0.35 and abs(cum.f40 - 0.85) <= 0.50:
        qpsk_notes.append(f"Cumulants f20={cum.f20:.3f}, f40={cum.f40:.3f} match 4-phase circular constellation.")
    if ph.var_phase_4th <= 0.60:
        qpsk_notes.append(f"4th-power phase variance is low ({ph.var_phase_4th:.3f}), indicating 90-deg phase state collapse.")

    if cum.f20 >= 0.50:
        qpsk_contra.append(f"High f20={cum.f20:.3f} contradicts 2D circular QPSK.")

    qpsk_total = 0.15 * qpsk_amp_score + 0.40 * qpsk_cum_score + 0.30 * qpsk_ph_score + 0.15 * cyc.periodicity_score
    if cum.f20 >= 0.50 or has_fsk_states:
        qpsk_total *= 0.25
    elif cum.f40 <= 0.25:
        # Lower score slightly if f40 is low, but retain as candidate under potential CFO
        qpsk_total *= 0.75
    if not has_digital_clock:
        qpsk_total *= 0.20

    scores[(ModulationFamily.PSK, 4)] = round(float(np.clip(qpsk_total, 0.0, 0.99)), 3)
    evidence[(ModulationFamily.PSK, 4)] = {
        "amplitude": qpsk_amp_score,
        "phase": qpsk_ph_score,
        "frequency": 0.5,
        "cumulants": qpsk_cum_score,
        "spectral": 0.5,
        "periodicity": cyc.periodicity_score,
    }
    notes[(ModulationFamily.PSK, 4)] = qpsk_notes
    contra[(ModulationFamily.PSK, 4)] = qpsk_contra

    # =========================================================================
    # 4. 8-PSK Evaluation
    # =========================================================================
    psk8_amp_score = float(np.clip(1.0 - (amp.norm_variance / 0.25), 0.05, 0.99))
    # Circular C20 approx 0, C40 approx 0.0, C42 approx 0.8
    psk8_cum_score = float(np.clip((1.0 - min(cum.f20 / 0.35, 1.0)) * 0.4 + (1.0 - min(cum.f40 / 0.30, 1.0)) * 0.4 + (1.0 - min(abs(cum.f42 - 0.8) / 0.7, 1.0)) * 0.2, 0.05, 0.99))
    psk8_ph_score = float(np.clip((1.0 - min(ph.var_phase_8th / 0.85, 1.0)) * 0.60 + min(ph.var_phase_4th / 0.6, 1.0) * 0.40, 0.05, 0.99))
    psk8_notes = []
    psk8_contra = []

    if has_fsk_states:
        psk8_contra.append("Bimodal instantaneous frequency indicates FSK rather than 8-PSK.")

    if not has_digital_clock:
        psk8_contra.append("Lack of digital symbol clock periodicity contradicts digital 8-PSK.")

    if ph.var_phase_8th >= 0.80:
        psk8_contra.append(f"Lack of 8th-power phase collapse (var_phase_8th={ph.var_phase_8th:.3f} >= 0.80) contradicts 8-PSK.")

    if ph.phase_inc_var < 0.02:
        psk8_contra.append("Very low phase increment variance indicates continuous FM/analog signal rather than 8-PSK.")

    if cum.f40 <= 0.25 and cum.f20 <= 0.30:
        psk8_notes.append(f"Low f40={cum.f40:.3f} and f20={cum.f20:.3f} distinguish 8-PSK from QPSK/BPSK.")
    if ph.var_phase_8th <= 0.80:
        psk8_notes.append(f"8th-power phase variance is low ({ph.var_phase_8th:.3f}), indicating 45-deg phase state collapse.")

    if cum.f40 >= 0.50 or (ph.var_phase_4th <= 0.50 and cum.f40 >= 0.30):
        psk8_contra.append("High f40 or low 4th-phase variance suggests QPSK or 16-QAM rather than 8-PSK.")
    if cum.f20 >= 0.50:
        psk8_contra.append(f"High f20={cum.f20:.3f} contradicts 8-PSK.")

    psk8_total = 0.15 * psk8_amp_score + 0.40 * psk8_cum_score + 0.30 * psk8_ph_score + 0.15 * cyc.periodicity_score
    if cum.f40 >= 0.50 or cum.f20 >= 0.50 or has_fsk_states or (ph.var_phase_4th <= 0.50 and cum.f40 >= 0.30) or ph.var_phase_8th >= 0.80 or ph.phase_inc_var < 0.02:
        psk8_total *= 0.25
    if not has_digital_clock:
        psk8_total *= 0.20

    scores[(ModulationFamily.PSK, 8)] = round(float(np.clip(psk8_total, 0.0, 0.99)), 3)
    evidence[(ModulationFamily.PSK, 8)] = {
        "amplitude": psk8_amp_score,
        "phase": psk8_ph_score,
        "frequency": 0.5,
        "cumulants": psk8_cum_score,
        "spectral": 0.5,
        "periodicity": cyc.periodicity_score,
    }
    notes[(ModulationFamily.PSK, 8)] = psk8_notes
    contra[(ModulationFamily.PSK, 8)] = psk8_contra

    # =========================================================================
    # 5. 16-QAM Evaluation
    # =========================================================================
    qam_kurt_match = float(np.clip(1.0 - min(abs(amp.kurtosis - (-0.70)) / 1.0, 1.0), 0.0, 1.0))
    qam_amp_score = float(np.clip(min(amp.norm_variance / 0.08, 1.0) * 0.40 + qam_kurt_match * 0.60, 0.05, 0.99))
    qam_cum_score = float(np.clip((1.0 - min(abs(cum.f40 - 0.65) / 0.4, 1.0)) * 0.5 + (1.0 - min(abs(cum.f42 - 0.55) / 0.4, 1.0)) * 0.5, 0.05, 0.99))
    qam_ph_score = 0.60
    qam_notes = []
    qam_contra = []

    if has_fsk_states:
        qam_contra.append("Bimodal instantaneous frequency indicates FSK rather than 16-QAM.")

    if not has_digital_clock:
        qam_contra.append("Lack of digital symbol clock periodicity contradicts digital 16-QAM.")

    if amp.kurtosis <= -0.10:
        qam_notes.append(f"Negative envelope excess kurtosis ({amp.kurtosis:.2f}) matches square 16-QAM multi-ring grid.")
    else:
        qam_contra.append(f"Positive envelope kurtosis ({amp.kurtosis:.2f}) is inconsistent with 16-QAM constellation.")

    if amp.norm_variance >= 0.08:
        qam_notes.append(f"Non-constant amplitude envelope (norm_variance={amp.norm_variance:.4f}) supports QAM.")

    if cum.f20 >= 0.50:
        qam_contra.append(f"High f20={cum.f20:.3f} contradicts circular/square QAM.")

    qam_total = 0.35 * qam_amp_score + 0.35 * qam_cum_score + 0.15 * qam_ph_score + 0.15 * cyc.periodicity_score
    if amp.kurtosis > 0.10 or has_fsk_states or cum.f20 >= 0.50:
        qam_total *= 0.25
    if not has_digital_clock:
        qam_total *= 0.20

    scores[(ModulationFamily.QAM, 16)] = round(float(np.clip(qam_total, 0.0, 0.99)), 3)
    evidence[(ModulationFamily.QAM, 16)] = {
        "amplitude": qam_amp_score,
        "phase": qam_ph_score,
        "frequency": 0.5,
        "cumulants": qam_cum_score,
        "spectral": 0.5,
        "periodicity": cyc.periodicity_score,
    }
    notes[(ModulationFamily.QAM, 16)] = qam_notes
    contra[(ModulationFamily.QAM, 16)] = qam_contra

    return ClassicalScores(
        scores=scores,
        evidence_breakdown=evidence,
        supporting_notes=notes,
        contradictions=contra,
    )
