"""Seeded semi-synthetic channel impairments; never a representative-capture substitute."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class ImpairmentRecipe:
    seed: int
    pulse_shape: str = "rrc"
    rolloff: float = .35
    samples_per_symbol: float = 3.7
    timing_drift_ppm: float = 20.
    sample_rate_offset_ppm: float = 15.
    cfo_cycles_per_sample: float = .01
    cfo_drift_cycles_per_sample2: float = 0.
    iq_gain_db: float = 0.
    iq_phase_deg: float = 0.
    dc: complex = 0j
    phase_noise_std: float = 0.
    quantization: str = "float32"
    clip: float = 1.
    multipath_fir: tuple[complex, ...] = (1+0j,)
    interferer_frequency: float | None = None
    interferer_amplitude: float = 0.
    burst_period: int = 0
    burst_on: int = 0

def rrc_taps(rolloff: float, sps: float, span: int = 10) -> np.ndarray:
    if not 0 <= rolloff <= 1 or sps <= 1: raise ValueError("rolloff must be [0,1], sps > 1")
    n = np.arange(-span*sps, span*sps+1) / sps; out = np.empty_like(n)
    for i,t in enumerate(n):
        if abs(t) < 1e-10: out[i] = 1-rolloff+4*rolloff/np.pi
        elif rolloff and abs(abs(t)-1/(4*rolloff)) < 1e-8: out[i] = rolloff/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/(4*rolloff))+(1-2/np.pi)*np.cos(np.pi/(4*rolloff)))
        else: out[i] = (np.sin(np.pi*t*(1-rolloff))+4*rolloff*t*np.cos(np.pi*t*(1+rolloff)))/(np.pi*t*(1-(4*rolloff*t)**2))
    return out / np.sqrt(np.sum(out*out))

def impair(symbols: np.ndarray, recipe: ImpairmentRecipe) -> tuple[np.ndarray, dict]:
    """Generate T1 samples independently from tests/test_synthesis.py's generators."""
    rng=np.random.default_rng(recipe.seed); symbols=np.asarray(symbols,np.complex64)
    positions=np.arange(int(np.ceil(symbols.size*recipe.samples_per_symbol)))/recipe.samples_per_symbol
    base=np.interp(positions,np.arange(symbols.size),symbols.real,left=0,right=0)+1j*np.interp(positions,np.arange(symbols.size),symbols.imag,left=0,right=0)
    if recipe.pulse_shape == "rrc": base=np.convolve(base,rrc_taps(recipe.rolloff,recipe.samples_per_symbol),mode="same")
    elif recipe.pulse_shape != "rectangular": raise ValueError("pulse_shape must be rrc or rectangular")
    pos=np.arange(base.size)*(1+(recipe.timing_drift_ppm+recipe.sample_rate_offset_ppm)*1e-6)
    signal=np.interp(pos,np.arange(base.size),base.real,left=0,right=0)+1j*np.interp(pos,np.arange(base.size),base.imag,left=0,right=0)
    n=np.arange(signal.size); signal*=np.exp(2j*np.pi*(recipe.cfo_cycles_per_sample*n+.5*recipe.cfo_drift_cycles_per_sample2*n*n))
    phase=np.cumsum(rng.normal(0,recipe.phase_noise_std,signal.size)); signal*=np.exp(1j*phase)
    gain=10**(recipe.iq_gain_db/20); i=signal.real*gain; q=signal.imag
    signal=(i+1j*(q*np.cos(np.deg2rad(recipe.iq_phase_deg))+i*np.sin(np.deg2rad(recipe.iq_phase_deg))))+recipe.dc
    signal=np.convolve(signal,np.asarray(recipe.multipath_fir,np.complex64),mode="same")
    if recipe.interferer_frequency is not None: signal+=recipe.interferer_amplitude*np.exp(2j*np.pi*recipe.interferer_frequency*n)
    if recipe.burst_period: signal*=np.where(n%recipe.burst_period < recipe.burst_on,1.,0.)
    signal=np.clip(signal.real,-recipe.clip,recipe.clip)+1j*np.clip(signal.imag,-recipe.clip,recipe.clip)
    if recipe.quantization == "int8": signal=(np.rint(signal.real*127).clip(-128,127).astype(np.int8).astype(np.float32)/127+1j*np.rint(signal.imag*127).clip(-128,127).astype(np.int8).astype(np.float32)/127)
    elif recipe.quantization == "int16": signal=(np.rint(signal.real*32767).clip(-32768,32767).astype(np.int16).astype(np.float32)/32767+1j*np.rint(signal.imag*32767).clip(-32768,32767).astype(np.int16).astype(np.float32)/32767)
    elif recipe.quantization != "float32": raise ValueError("quantization must be float32, int8, or int16")
    logged=asdict(recipe); logged["dc"]=[recipe.dc.real,recipe.dc.imag]; logged["multipath_fir"]=[[x.real,x.imag] for x in recipe.multipath_fir]; logged.update(tier="T1",representativeness="NOT_REPRESENTATIVE")
    return np.ascontiguousarray(signal,dtype=np.complex64),logged

def estimate_iq_gain_db(samples: np.ndarray) -> float:
    samples=np.asarray(samples); return float(20*np.log10(np.std(samples.real)/np.std(samples.imag)))
