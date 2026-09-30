import pytest
import os
from pathlib import Path
from signal_analysis.pipeline import run_full_pipeline, PipelineStageStatus
from signal_analysis.loaders import WavReader

def test_demo_mode_fixtures():
    # Verify the demo mode fixtures produce expected pipeline status
    base_dir = Path(__file__).parent.parent / "fixtures" / "data_"
    if not base_dir.exists():
        pytest.skip("Demo fixtures not generated")

    fixtures = {
        "clean_qpsk.wav": PipelineStageStatus.COMPLETED,
        "concatenated.wav": PipelineStageStatus.COMPLETED,
        "low_snr_qpsk.wav": PipelineStageStatus.COMPLETED,
        "ofdm_out_of_scope.wav": PipelineStageStatus.NOT_ATTEMPTED,
        "real_valued_gate.wav": PipelineStageStatus.NOT_ATTEMPTED,
        "qam_clean.wav": PipelineStageStatus.COMPLETED,
        "qam_low_snr.wav": PipelineStageStatus.COMPLETED,
        "qam_concatenated.wav": PipelineStageStatus.COMPLETED,
        # The native beta profile includes high-order square QAM acquisition.
        "qam_unsupported_order.wav": PipelineStageStatus.COMPLETED,
        # CFO-tolerant estimation now passes the correct candidate to the native receiver.
        "qam_cfo_capture.wav": PipelineStageStatus.COMPLETED
    }

    for fname, expected_status in fixtures.items():
        path = base_dir / fname
        if not path.exists():
            continue
        
        mode = "stereo_real" if "real_valued" in fname else "stereo_iq"
        recording = WavReader(str(path), mode=mode).read()
        res = run_full_pipeline(recording)
        
        # Check sync_status directly matches expectation
        assert res.sync_status == expected_status, f"{fname} got sync_status {res.sync_status}, expected {expected_status}"
