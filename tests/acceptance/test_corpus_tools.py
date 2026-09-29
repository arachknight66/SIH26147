from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from tools.corpus.evaluate import clopper_pearson, minimum_n_zero_failure, _audit, FirewallError, evaluate
from tools.corpus.impair import ImpairmentRecipe, impair, estimate_iq_gain_db
from tools.corpus.validate_manifest import ManifestError, split_for, validate
from tools.corpus.ingest import production_request
from tools.corpus.evaluate import _prediction
from signal_analysis.workflow import run_production_analysis

def test_committed_t2_manifest_record_validates_without_local_binary():
    """The committed record is provenance only; the ignored binary is optional in CI."""
    captures = validate(Path('corpus/manifest.json'))['captures']
    assert len(captures) == 1
    record = captures[0]
    assert record['tier'] == 'T2'
    assert record['format'] == 'SIGMF'
    assert record['split'] in {'calibration', 'heldout'}
    assert record['sha256'] == record['import_config']['metadata_sidecar_sha256']
    assert len(record['import_config']['data_sidecar_sha256']) == 64

def test_unknown_truth_is_not_misreported_as_an_unsupported_negative(tmp_path):
    """Independent family truth is required before a capture enters either metric."""
    manifest = json.loads(Path('corpus/manifest.json').read_text())
    record = manifest['captures'][0].copy()
    record['path'] = 'missing.sigmf-meta'  # no production work is needed for this filtering check
    record['truth_ref'] = 'corpus/truth.json'
    manifest['captures'][0] = record
    corpus = tmp_path / 'corpus'; corpus.mkdir()
    manifest_path = corpus / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    (corpus / 'truth.json').write_text(json.dumps({'schema_version': 1, 'captures': {
        record['capture_id']: json.loads(Path('corpus/truth/truth.json').read_text())['captures'][record['capture_id']]
    }}))
    report = evaluate(manifest_path, tier='T2', split='calibration', output_dir=tmp_path / 'reports', allow_missing=True)
    assert report['metrics']['family_accuracy']['n'] == 0
    assert report['metrics']['high_confidence_wrong_unsupported']['n'] == 0

def test_manifest_rejects_hash_and_nonhash_split(tmp_path):
    salt='x'*16; split,digest=split_for('cap',salt)
    record={'capture_id':'cap','path':'data/cap.iq','sha256':'x'*64,'tier':'T2','source_description':'x','license':{'redistribution_status':'UNKNOWN'},'acquisition':{},'center_frequency_hz':{'value':None,'status':'MISSING'},'sample_rate_hz':{'value':None,'status':'MISSING'},'format':'RAW_IQ','import_config':{'dtype':'complex64','iq_order':'IQ','endian':'little'},'duration_seconds':0,'split':split,'split_assignment':{'algorithm':'sha256','salt_id':'v1','digest':digest,'rule':'first_hex_nibble_lt_8_calibration'},'truth_ref':'corpus/truth/truth.json','notes':''}
    path=tmp_path/'manifest.json'; path.write_text(json.dumps({'schema_version':1,'split_salt':salt,'captures':[record]}))
    with pytest.raises(ManifestError): validate(path)
    record['sha256']='0'*64; record['split']='heldout' if split=='calibration' else 'calibration'; path.write_text(json.dumps({'schema_version':1,'split_salt':salt,'captures':[record]}))
    with pytest.raises(ManifestError): validate(path)

def test_known_answer_intervals():
    assert clopper_pearson(0, 10) == pytest.approx([0, .308497])
    assert clopper_pearson(10, 10) == pytest.approx([.691503, 1])
    assert minimum_n_zero_failure(.01) == 368

def test_firewall_refuses_heldout_tuning(tmp_path):
    with pytest.raises(FirewallError): _audit(tmp_path, ['heldout-id'], 'heldout', 'abc', tuning=True)

def test_t1_determinism_and_iq_statistic():
    symbols=np.exp(2j*np.pi*np.arange(200)/4).astype(np.complex64); recipe=ImpairmentRecipe(seed=4,pulse_shape='rectangular',iq_gain_db=6,cfo_cycles_per_sample=0,quantization='int8',clip=10)
    a,log_a=impair(symbols,recipe); b,log_b=impair(symbols,recipe); measured,_=impair(symbols,ImpairmentRecipe(seed=4,pulse_shape='rectangular',iq_gain_db=6,cfo_cycles_per_sample=0,clip=10)); baseline,_=impair(symbols,ImpairmentRecipe(seed=4,pulse_shape='rectangular',cfo_cycles_per_sample=0,clip=10))
    assert a.tobytes()==b.tobytes() and log_a==log_b and log_a['tier']=='T1'
    assert estimate_iq_gain_db(measured)-estimate_iq_gain_db(baseline)==pytest.approx(6,abs=.1)

def test_noise_and_silence_never_have_synthetic_high_confidence_claims():
    rows=[{'status':'UNKNOWN','score':0.0},{'status':'AMBIGUOUS','score':.99}]
    assert not any(r['status']=='CANDIDATE' and r['score']>=.8 for r in rows)

def _raw_record(path: Path, capture_id: str) -> dict:
    digest=__import__('hashlib').sha256(path.read_bytes()).hexdigest()
    return {'capture_id':capture_id,'path':str(path),'sha256':digest,'format':'RAW_IQ','import_config':{'dtype':'complex64','iq_order':'IQ','endian':'little'},'sample_rate_hz':{'value':48000.,'status':'KNOWN'},'center_frequency_hz':{'value':None,'status':'MISSING'}}

def test_rename_invariance_and_silence_negative_control_through_production_ingestion(tmp_path):
    # Truth/IDs are not supplied to run_production_analysis; identical raw bytes remain identical after rename.
    samples=np.zeros(2048,np.complex64); first=tmp_path/'opaque-a.iq'; second=tmp_path/'different-label.iq'; samples.tofile(first); second.write_bytes(first.read_bytes())
    a=run_production_analysis(production_request(_raw_record(first,'a'),Path('/'))); b=run_production_analysis(production_request(_raw_record(second,'b'),Path('/')))
    assert _prediction(a)==_prediction(b)
    assert not (_prediction(a)['status']=='CANDIDATE' and _prediction(a)['score']>=.8)
