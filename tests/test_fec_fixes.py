
def test_fix3_rs_reencode_mismatch(monkeypatch):
    from signal_analysis.fec_reed_solomon import ReedSolomon
    from signal_analysis.models import Diagnostic, Severity
    
    rs = ReedSolomon(15, 11)
    msg = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    encoded = rs.encode(msg)
    
    # Introduce 1 error, should be correctable
    corrupted = list(encoded)
    corrupted[0] ^= 0x55
    
    # Monkeypatch calc_syndromes to return all zeros during verification pass ONLY
    original_calc = rs.calc_syndromes
    call_count = 0
    def fake_calc(codeword):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            # First time (before decoding) it computes syndromes normally
            # Second time (verification pass) we pretend it succeeded
            return [0] * (rs.n - rs.k)
        return original_calc(codeword)
        
    monkeypatch.setattr(rs, "calc_syndromes", fake_calc)
    
    def fake_forney(syndromes, err_loc, roots):
        return [0x99] # wrong magnitude
    monkeypatch.setattr(rs, "forney_magnitudes", fake_forney)
    
    # And mock find_roots to return [0]
    monkeypatch.setattr(rs, "find_roots_chien", lambda elp: [0])
    
    dec_msg, err_count, success, diags = rs.decode(corrupted)
    
    # With the fake calc_syndromes, the syndrome check passes (0)
    # But re-encoding will reveal that the corrected parity doesn't match the re-encoded parity
    assert not success
    assert any(d.code == "RS_REENCODE_MISMATCH" for d in diags)
