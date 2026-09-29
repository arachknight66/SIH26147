"""Exact rate intervals and conservative gate verdicts."""
from __future__ import annotations
import math
from dataclasses import dataclass
from scipy.stats import beta
@dataclass(frozen=True)
class RateStats:
    n:int; k:int; rate:float|None; clopper_pearson_95:tuple[float,float]|None; one_sided_upper_95:float|None; jeffreys_95:tuple[float,float]|None
def rate_stats(k:int,n:int)->RateStats:
    if not n:return RateStats(0,k,None,None,None,None)
    cp=(0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k)))
    one=1. if k==n else float(beta.ppf(.95,k+1,n-k))
    jeff=(float(beta.ppf(.025,k+.5,n-k+.5)),float(beta.ppf(.975,k+.5,n-k+.5)))
    return RateStats(n,k,k/n,cp,one,jeff)
def minimum_n(target:float,two_sided:bool=True)->int:return math.ceil(math.log(.025 if two_sided else .05)/math.log(1-target))
def verdict(stats:RateStats,target:float)->str:
    if not stats.n:return "NOT_MEASURABLE"
    if stats.k and stats.rate is not None and stats.rate>target:return "FAIL"
    if stats.clopper_pearson_95[0]>target:return "FAIL"
    if stats.one_sided_upper_95 < target:return "PASS"
    return "INSUFFICIENT_POWER"
