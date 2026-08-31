with open('signal_analysis/models.py', 'a') as f:
    f.write("""
@dataclass(frozen=True)
class SynchronizationResult:
    cfo_estimate: float
    cfo_unit: str  # "Hz" or "cycles/sample"
    timing_offset_fractional_symbols: float
    symbol_clock_locked: bool
    carrier_locked: bool
    lock_quality_metric: float
    evm_percent: float
    diagnostics: List[Diagnostic]

@dataclass(frozen=True)
class DemodulationResult:
    hard_bits: np.ndarray  # uint8
    soft_llrs: np.ndarray  # float32, positive means bit=1
    bits_per_symbol: int
    symbol_decisions: np.ndarray  # complex64
    sync_result: SynchronizationResult
    source_hypothesis_label: str
    hypothesis_confirmed: bool
""")
print("Appended models.py")
