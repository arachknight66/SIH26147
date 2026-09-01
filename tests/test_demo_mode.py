import pytest
import os
from pathlib import Path
from signal_analysis.pipeline import run_full_pipeline, PipelineStageStatus
from signal_analysis.loaders import WavReader

def test_demo_mode_fixtures():
    # Verify the demo mode fixtures produce expected pipeline status
    base_dir = Path(__file__).parent.parent / "fixtures" / "demo"
    if not base_dir.exists():
        pytest.skip("Demo fixtures not generated")

    fixtures = {
        "demo_clean_qpsk.wav": PipelineStageStatus.COMPLETED,
        "demo_concatenated.wav": PipelineStageStatus.COMPLETED,
        "demo_low_snr_qpsk.wav": PipelineStageStatus.COMPLETED,
        "demo_ofdm_out_of_scope.wav": PipelineStageStatus.NOT_ATTEMPTED,
        "demo_real_valued_gate.wav": PipelineStageStatus.NOT_ATTEMPTED,
        "demo_qam_clean.wav": PipelineStageStatus.COMPLETED,
        "demo_qam_low_snr.wav": PipelineStageStatus.COMPLETED,
        "demo_qam_concatenated.wav": PipelineStageStatus.COMPLETED,
        # The unsupported 64-QAM gets mislabeled confidently as 16-QAM and goes through to sync (where EVM is high but sync_status still completes)
        "demo_qam_unsupported_order.wav": PipelineStageStatus.COMPLETED,
        # 16-QAM with CFO gets confidently mislabeled as QPSK, which fails sync because QPSK costas loop cannot lock it.
        "demo_qam_cfo_capture.wav": PipelineStageStatus.FAILED
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
