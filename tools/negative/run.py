"""Resumable fixed negative suite runner."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, time, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from signal_analysis.pipeline import run_full_pipeline
from tools.corpus.evaluate import _audit
from .claims import evaluate_claims
from .generators import STRATA, generate, spec
from .report import build, write
ROOT=Path(__file__).resolve().parents[2]
def _one(args):
    suite,stratum,index=args; item=spec(suite,stratum,index); started=time.perf_counter()
    try:
        result=run_full_pipeline(generate(item)); claims=evaluate_claims(result)
        top=result.top_hypothesis
        return {"suite":suite,"stratum":stratum,"index":index,"entropy":item.entropy,"levels":claims.levels,"uncoded_success":claims.uncoded_success,"top_label":None if not top else top.label,"top_score":None if not top else top.score,"top_status":None if not top else top.status.name,"sync_status":result.sync_status.name,"fec_status":result.fec_status.name,"framing_status":result.framing_status.name,"error":None,"seconds":time.perf_counter()-started}
    except Exception:
        text=traceback.format_exc(); return {"suite":suite,"stratum":stratum,"index":index,"entropy":item.entropy,"levels":{x:False for x in evaluate_claims.__globals__["LEVELS"]},"error":{"traceback_sha256":hashlib.sha256(text.encode()).hexdigest()},"seconds":time.perf_counter()-started}
def git_state():
    try:return {"commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"dirty":bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip())}
    except Exception:return {"commit":"UNAVAILABLE","dirty":True}
def run(suite:str,workers:int=1,resume=False):
    each=25 if suite=="smoke" else 1250; out=ROOT/"build/negative"; out.mkdir(parents=True,exist_ok=True); jsonl=out/f"{suite}.jsonl"; done=set()
    if resume and jsonl.exists(): done={(r["stratum"],r["index"]) for r in map(json.loads,jsonl.read_text().splitlines())}
    tasks=[(suite,s,i) for s in STRATA for i in range(each) if (s,i) not in done]
    audit_count=(_audit(ROOT,[f"negative-{suite}-{s}-{i}" for _,s,i in tasks],"heldout",hashlib.sha256(f"negative-v1-{suite}".encode()).hexdigest())+1) if suite=="release" else 0
    iterator=map(_one,tasks) if workers==1 else ProcessPoolExecutor(max_workers=workers).map(_one,tasks)
    with jsonl.open("a",encoding="utf-8") as f:
        for row in iterator:f.write(json.dumps(row,sort_keys=True)+"\n");f.flush()
    report=build(jsonl,suite,each,audit_count,git_state()); write(report,out/f"{suite}_report.json"); return report
def main():
    p=argparse.ArgumentParser();p.add_argument("--suite",choices=["smoke","release"],required=True);p.add_argument("--workers",type=int,default=1);p.add_argument("--resume",action="store_true");a=p.parse_args();r=run(a.suite,a.workers,a.resume);print(json.dumps({"suite":a.suite,"n":r["pooled"]["n"],"l5":r["pooled"]["levels"]["L5_confirmed_frame"]},indent=2))
if __name__=="__main__":main()
