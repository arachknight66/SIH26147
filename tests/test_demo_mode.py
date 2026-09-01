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
        "demo_real_valued_gate.wav": PipelineStageStatus.NOT_ATTEMPTED
    }

    for fname, expected_status in fixtures.items():
        path = base_dir / fname
        if not path.exists():
            continue
        
        mode = "stereo_real" if "real_valued" in fname else "stereo_iq"
        recording = WavReader(str(path), mode=mode).read()
        res = run_full_pipeline(recording)
        
        # We check sync_status since NOT_ATTEMPTED happens early (e.g. at hypothesis for OFDM/real)
        if expected_status == PipelineStageStatus.COMPLETED:
            assert res.sync_status == PipelineStageStatus.COMPLETED, f"{fname} failed sync_status"
        else:
            assert res.sync_status == PipelineStageStatus.NOT_ATTEMPTED, f"{fname} should not have run sync"
