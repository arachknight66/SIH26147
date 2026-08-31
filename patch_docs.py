import ast
import glob
import os

DOCSTRINGS = {
    "extract_amplitude_features": '"""\n    Extract amplitude-based statistical features.\n\n    Parameters\n    ----------\n    samples : np.ndarray\n        Complex baseband samples.\n\n    Returns\n    -------\n    AmplitudeFeatures\n        Extracted amplitude features (variance, kurtosis, skewness).\n    """',
    "extract_phase_features": '"""\n    Extract phase-based statistical features.\n\n    Parameters\n    ----------\n    samples : np.ndarray\n        Complex baseband samples.\n\n    Returns\n    -------\n    PhaseFeatures\n        Extracted phase features.\n    """',
    "extract_frequency_features": '"""\n    Extract frequency-based statistical features.\n\n    Parameters\n    ----------\n    samples : np.ndarray\n        Complex baseband samples.\n\n    Returns\n    -------\n    FrequencyFeatures\n        Extracted frequency features.\n    """',
    "extract_cumulant_features": '"""\n    Extract higher-order cumulant features.\n\n    Parameters\n    ----------\n    samples : np.ndarray\n        Complex baseband samples.\n\n    Returns\n    -------\n    CumulantFeatures\n        C20, C21, C40, C41, C42 cumulants.\n    """',
    "extract_spectral_features": '"""\n    Extract spectral symmetry and bandwidth features.\n\n    Parameters\n    ----------\n    recording : SignalRecording\n        Signal recording.\n\n    Returns\n    -------\n    SpectralFeatures\n        Extracted spectral features.\n    """',
    "extract_cyclostationary_features": '"""\n    Extract cyclostationary features for OFDM detection.\n\n    Parameters\n    ----------\n    recording : SignalRecording\n        Signal recording.\n\n    Returns\n    -------\n    CyclostationaryFeatures\n        Extracted features.\n    """',
    "estimate_rate_transition_energy": '"""\n    Estimate symbol rate using zero-crossing/transition energy.\n\n    Parameters\n    ----------\n    samples : np.ndarray\n        Complex or real samples.\n\n    Returns\n    -------\n    Tuple[Optional[float], float]\n        Estimated symbol rate fraction and confidence.\n    """',
    "estimate_rate_squared_magnitude": '"""\n    Estimate symbol rate using squared magnitude spectrum.\n\n    Parameters\n    ----------\n    samples : np.ndarray\n        Complex samples.\n\n    Returns\n    -------\n    Tuple[Optional[float], float]\n        Estimated symbol rate fraction and confidence.\n    """',
    "estimate_rate_autocorrelation": '"""\n    Estimate symbol rate using cyclostationary autocorrelation.\n\n    Parameters\n    ----------\n    samples : np.ndarray\n        Complex samples.\n\n    Returns\n    -------\n    Tuple[Optional[float], float]\n        Estimated symbol rate fraction and confidence.\n    """',
    "rrc_filter": '"""\n    Generate a Root Raised Cosine (RRC) filter impulse response.\n\n    Parameters\n    ----------\n    sps : float\n        Samples per symbol.\n    alpha : float\n        Roll-off factor (default 0.35 is standard for satellite/terrestrial links).\n    length : int\n        Filter length in symbols (default 8).\n\n    Returns\n    -------\n    np.ndarray\n        Filter coefficients.\n    """',
    "recover_carrier_costas": '"""\n    Recover carrier phase and frequency using a Costas loop.\n\n    Parameters\n    ----------\n    symbols : np.ndarray\n        Input symbols.\n    modulation : str\n        Target modulation scheme (e.g. BPSK, QPSK).\n\n    Returns\n    -------\n    Tuple[np.ndarray, bool, float]\n        Phase-corrected symbols, lock status, and lock quality metric.\n    \n    Notes\n    -----\n    Loop bandwidth and damping constants are hardcoded for typical SNR conditions.\n    """',
    "recover_timing_fsk": '"""\n    Recover timing for FSK signals.\n\n    Parameters\n    ----------\n    samples : np.ndarray\n        Input samples.\n    sps : float\n        Samples per symbol.\n\n    Returns\n    -------\n    Tuple[bool, float, np.ndarray]\n        Lock status, symbol timing, and resampled symbols.\n    """',
    "fsk_dual_correlator": '"""\n    Perform dual-branch correlation for FSK demodulation.\n\n    Parameters\n    ----------\n    samples : np.ndarray\n        Input samples.\n    symbol_indices : np.ndarray\n        Indices of symbol centers.\n    f0 : float\n        Mark frequency.\n    f1 : float\n        Space frequency.\n\n    Returns\n    -------\n    Tuple[np.ndarray, np.ndarray, np.ndarray, float]\n        Hard decisions, soft LLRs, symbol powers, and EVM estimate.\n    """',
    "evaluate_and_rank_hypotheses": '"""\n    Evaluate and rank modulation hypotheses.\n\n    Parameters\n    ----------\n    recording : SignalRecording\n        Signal recording.\n    fv : ModulationFeatureVector\n        Extracted feature vector.\n\n    Returns\n    -------\n    List[ModulationHypothesis]\n        Ranked list of hypotheses.\n    """',
    "check_temporal_consistency": '"""\n    Check temporal consistency of signal features.\n\n    Parameters\n    ----------\n    recording : SignalRecording\n        Signal recording.\n    config : Dict[str, Any]\n        Configuration dictionary.\n\n    Returns\n    -------\n    Tuple[float, Optional[Diagnostic]]\n        Consistency score and diagnostic if inconsistent.\n    """',
    "search_interleaver_hypotheses": '"""\n    Search for block interleaver dimensions.\n\n    Parameters\n    ----------\n    demod_result : DemodulationResult\n        Demodulation output containing hard bits.\n    config : Optional[Dict[str, Any]]\n        Configuration for test dimensions.\n\n    Returns\n    -------\n    List[DeinterleaverHypothesis]\n        Ranked interleaver hypotheses.\n    """',
    "compute_classical_scores": '"""\n    Compute scores for classical modulations based on features.\n\n    Parameters\n    ----------\n    feature_vector : ModulationFeatureVector\n        Extracted features.\n\n    Returns\n    -------\n    Dict[str, ClassScore]\n        Dictionary mapping modulation labels to their classification scores.\n    """',
    "decode_reed_solomon": '"""\n    Decode a concatenated/deinterleaved block using Reed-Solomon.\n\n    Parameters\n    ----------\n    deint : DeinterleavingResult\n        Input deinterleaved bits.\n    n : int\n        RS block length (bytes).\n    k : int\n        RS message length (bytes).\n\n    Returns\n    -------\n    FECDecodeResult\n        Decoding success and diagnostics.\n    """',
    "run_cli": '"""\n    Entry point for the CLI.\n\n    Parses arguments, loads the signal, and runs the pipeline.\n    """'
}

def patch_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content)
    lines = content.split('\n')
    
    inserts = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and not ast.get_docstring(node):
            if node.name in DOCSTRINGS:
                # Add docstring just after the function definition
                # Function def might span multiple lines if arguments are wrapped
                # Find the colon that ends the function def
                end_lineno = node.body[0].lineno - 1
                while not lines[end_lineno - 1].rstrip().endswith(':'):
                    end_lineno -= 1
                    
                # We will insert at end_lineno
                col_offset = node.body[0].col_offset
                indent = ' ' * col_offset
                doc = DOCSTRINGS[node.name]
                doc_indented = '\n'.join([(indent + line if line else '') for line in doc.split('\n')])
                inserts.append((end_lineno, doc_indented))
                
    if not inserts:
        return
        
    inserts.sort(key=lambda x: x[0], reverse=True)
    for lineno, doc in inserts:
        lines.insert(lineno, doc)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

for p in glob.glob('signal_analysis/*.py'):
    patch_file(p)
print("Docstrings patched.")
