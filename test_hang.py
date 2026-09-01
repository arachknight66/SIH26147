import numpy as np
import time
from signal_analysis.framing import assemble_frames
from signal_analysis.models import HeaderMatch, SyncWordPattern

bits = np.random.randint(0, 2, 100000, dtype=np.uint8)
header_matches = [
    HeaderMatch(
        pattern=SyncWordPattern("HDLC_FLAG", [0,1,1,1,1,1,1,0], "", "", ""),
        bit_offset=i,
        hamming_distance=0,
        match_confidence=1.0,
        periodicity_consistent=False
    )
    for i in range(0, 10000, 500) # 20 matches
]

print(f"Bits: {len(bits)}, Matches: {len(header_matches)}")
t0 = time.time()
try:
    # Just one header match, very short bitstream
    assemble_frames(bits[:1000], header_matches[:1])
except Exception as e:
    import traceback
    traceback.print_exc()
t1 = time.time()
print(f"Time: {t1-t0:.2f}s")
