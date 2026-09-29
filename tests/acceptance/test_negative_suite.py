from __future__ import annotations
import numpy as np
import pytest
from tools.negative.generators import STRATA, bytes_hash, generate, spec
from tools.negative.stats import minimum_n, rate_stats, verdict

def test_exact_interval_known_answers():
    tenk=rate_stats(0,10000); assert tenk.one_sided_upper_95==pytest.approx(2.9953e-4,rel=1e-4); assert tenk.clopper_pearson_95[1]==pytest.approx(3.6882e-4,rel=1e-4)
    assert rate_stats(0,299).one_sided_upper_95==pytest.approx(.00997,rel=2e-3)
    assert rate_stats(0,368).clopper_pearson_95[1] < .01
    assert minimum_n(.01)==368

def test_seed_regeneration_hash_and_recording_metadata():
    assert bytes_hash("release",STRATA[0],3)==bytes_hash("release",STRATA[0],3)
    assert generate(spec("release","S4_colored_real",1)).semantic_type=="mono_real"
    assert generate(spec("release","S1_white_noise",1)).sample_rate_hz.value is None

def test_faulted_failure_is_fail_not_dropped():
    s=rate_stats(2,100); assert verdict(s,.01)=="FAIL"
