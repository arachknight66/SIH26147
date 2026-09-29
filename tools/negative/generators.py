"""Bit-identical seeded synthetic negative-window generators (not test_synthesis)."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from signal_analysis.models import MetadataStatus, MetadataValue, SignalRecording, SourceFormat
from signal_analysis.correlation import BUILTIN_SYNC_WORDS
SUITE_VERSION=1
STRATA=("S1_white_noise","S2_silence","S3_short","S4_colored_real","S5_interference","S6_unsupported","S7_low_snr_lookalike","S8_random_framed")
@dataclass(frozen=True)
class WindowSpec: suite:str; stratum:str; index:int; entropy:tuple[int,int,int]
def spec(suite:str,stratum:str,index:int)->WindowSpec:return WindowSpec(suite,stratum,index,(SUITE_VERSION,STRATA.index(stratum),index))
def _rng(item:WindowSpec): return np.random.Generator(np.random.PCG64(np.random.SeedSequence(item.entropy)))
def _recording(x:np.ndarray,semantic="complex_iq",known=True)->SignalRecording:
    return SignalRecording(np.ascontiguousarray(x,dtype=np.complex64),SourceFormat.RAW_IQ,"complex64",semantic,MetadataValue(1_000_000. if known else None,"negative-suite",MetadataStatus.KNOWN if known else MetadataStatus.MISSING),MetadataValue(None,"negative-suite",MetadataStatus.MISSING),{"negative_suite":True},[])
def _qpsk(bits,sps=4):
    pairs=bits[:len(bits)//2*2].reshape(-1,2); sym=(2*pairs[:,0]-1+1j*(2*pairs[:,1]-1))/np.sqrt(2); return np.repeat(sym,sps)
def generate(item:WindowSpec)->SignalRecording:
    rng=_rng(item); s=item.stratum; length=(512,2048,8192,65536)[item.index%4]
    if s=="S1_white_noise": x=(rng.normal(size=length)+1j*rng.normal(size=length))*10**((item.index%5-2)/2); return _recording(x,known=item.index%2==0)
    if s=="S2_silence":
        mode=item.index%3; x=np.zeros(length) if mode==0 else (rng.normal(0,1e-7,length)+1j*rng.normal(0,1e-7,length) if mode==1 else np.full(length,.2+.1j)); return _recording(x)
    if s=="S3_short": return _recording(rng.normal(size=(32,128,255,512)[item.index%4])+1j*rng.normal(size=(32,128,255,512)[item.index%4]))
    if s=="S4_colored_real":
        white=rng.normal(size=length); fir=rng.normal(size=9); x=np.convolve(white,fir/fir.std(),"same"); return _recording(x,"mono_real",known=False)
    if s=="S5_interference":
        n=np.arange(length); x=np.exp(2j*np.pi*(.03+.0000005*n)*n)+.5*np.exp(2j*np.pi*.17*n); x[::97]+=4; return _recording(x)
    if s=="S6_unsupported":
        n=np.arange(length); fft=64; body=(rng.normal(size=fft)+1j*rng.normal(size=fft)); ofdm=np.fft.ifft(body); x=np.tile(np.r_[ofdm[-16:],ofdm],length//80+1)[:length]; return _recording(x)
    bits=rng.integers(0,2,length//4*2,dtype=np.uint8); x=_qpsk(bits)
    if s=="S7_low_snr_lookalike": x=x+np.sqrt(10**(-(5+item.index%9)/10)/2)*(rng.normal(size=x.size)+1j*rng.normal(size=x.size)); return _recording(x)
    # S8 deliberately rejects every exact built-in sync pattern before modulation.
    for pattern in BUILTIN_SYNC_WORDS:
        windows=np.lib.stride_tricks.sliding_window_view(bits,pattern.bit_pattern.size)
        while np.any(np.all(windows==pattern.bit_pattern,axis=1)):
            for pos in np.where(np.all(windows==pattern.bit_pattern,axis=1))[0]: bits[pos]^=1
            windows=np.lib.stride_tricks.sliding_window_view(bits,pattern.bit_pattern.size)
    return _recording(x)
def bytes_hash(suite:str,stratum:str,count:int)->str:
    h=hashlib.sha256()
    for i in range(count): h.update(generate(spec(suite,stratum,i)).samples.tobytes())
    return h.hexdigest()
