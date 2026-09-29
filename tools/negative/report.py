"""Compute immutable negative-suite reports from checkpoint JSONL only."""
from __future__ import annotations
import json, platform
from collections import defaultdict
from pathlib import Path
import numpy, scipy
from .claims import LEVELS
from .generators import STRATA, bytes_hash
from .stats import rate_stats, verdict, minimum_n
SCOPE="Negatives here are SYNTHETIC (T0/T1 provenance). The bound applies to THIS generative distribution only and does not certify behavior on real RF."
def build(jsonl:Path,suite:str,count:int,invocations:int,git:dict)->dict:
    rows=[json.loads(x) for x in jsonl.read_text().splitlines() if x]; report={"schema_version":1,"suite":suite,"scope_of_claim":SCOPE,"environment":{"python":platform.python_version(),"numpy":numpy.__version__,"scipy":scipy.__version__},"release_suite_invocation_count":invocations,"git":git,"t2_real_rf_negatives":{"n":0,"pooled_with_synthetic":False},"strata":{},"pooled":{},"minimum_n_per_stratum_two_sided":minimum_n(.01)}
    for group,selected in [(s,[r for r in rows if r["stratum"]==s]) for s in STRATA]+[("pooled",rows)]:
        item={"n":len(selected),"window_bytes_sha256":None if group=="pooled" else bytes_hash(suite,group,count),"levels":{}}
        for level in LEVELS:
            stats=rate_stats(sum(bool(r["levels"].get(level)) for r in selected),len(selected)); item["levels"][level]={"n":stats.n,"k":stats.k,"rate":stats.rate,"clopper_pearson_95":stats.clopper_pearson_95,"one_sided_upper_95":stats.one_sided_upper_95,"jeffreys_95":stats.jeffreys_95,"verdict":verdict(stats,.01)}
        if group=="pooled": report["pooled"]=item
        else: report["strata"][group]=item
    for level in LEVELS:
        values=[report["strata"][s]["levels"][level]["verdict"] for s in STRATA]; report["pooled"]["levels"][level]["worst_stratum_verdict"]="FAIL" if "FAIL" in values else "INSUFFICIENT_POWER" if "INSUFFICIENT_POWER" in values else "PASS" if all(x=="PASS" for x in values) else "NOT_MEASURABLE"
    return report
def write(report:dict,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2,sort_keys=True)); md=path.with_suffix('.md'); md.write_text(f"# Negative suite {report['suite']}\n\n{SCOPE}\n\n| Level | pooled k/n | worst verdict |\n|---|---:|---|\n"+"\n".join(f"| {l} | {report['pooled']['levels'][l]['k']}/{report['pooled']['levels'][l]['n']} | {report['pooled']['levels'][l]['worst_stratum_verdict']} |" for l in LEVELS)+"\n")
