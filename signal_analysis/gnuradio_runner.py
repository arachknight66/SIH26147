"""Standalone GNU Radio flowgraph runner; execute with a GNU Radio Python."""

from __future__ import annotations

import argparse
from fractions import Fraction
import math
from pathlib import Path
import sys


# One complex sample is eight bytes.  These buffers are large enough to avoid
# scheduler churn on capture files while staying modest for a desktop runtime.
_BUFFER_ITEMS = 131_072


def _tune_buffer(block, *, output: bool) -> None:
    """Request bounded, high-throughput GNU Radio scheduler buffers.

    The methods are part of GNU Radio's basic-block API, but older builds can
    reject a request for an individual block.  The graph remains valid with
    its own defaults in that case.
    """
    try:
        if output:
            block.set_min_output_buffer(_BUFFER_ITEMS)
            block.set_max_output_buffer(_BUFFER_ITEMS * 2)
        else:
            block.set_min_input_buffer(_BUFFER_ITEMS)
            block.set_max_input_buffer(_BUFFER_ITEMS * 2)
    except (AttributeError, RuntimeError):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="SIH26147 GNU Radio raw CF32 preprocessor")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-rate", type=float, required=True)
    parser.add_argument("--output-rate", type=float, required=True)
    parser.add_argument("--input-format", choices=("cf32_le", "ci16_le"), default="cf32_le")
    parser.add_argument("--frequency-shift", type=float, default=0.0)
    parser.add_argument("--lowpass-cutoff", type=float)
    args = parser.parse_args()

    # This runner lives next to ``signal_analysis/gnuradio.py``.  Remove that
    # directory from module resolution so the system GNU Radio package, rather
    # than this adapter module, supplies ``gnuradio.blocks``.
    runner_directory = Path(__file__).resolve().parent
    sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != runner_directory]
    from gnuradio import blocks, filter, gr
    from gnuradio.filter import firdes

    flowgraph = gr.top_block("sih26147_preprocess")
    if args.input_format == "cf32_le":
        source = blocks.file_source(gr.sizeof_gr_complex, args.input, False)
        current = source
    else:
        source = blocks.file_source(gr.sizeof_short, args.input, False)
        converter = blocks.interleaved_short_to_complex(False, False, 1.0)
        flowgraph.connect(source, converter)
        current = converter
    _tune_buffer(source, output=True)
    if args.frequency_shift:
        rotator = blocks.rotator_cc(2.0 * math.pi * args.frequency_shift / args.input_rate)
        flowgraph.connect(current, rotator)
        current = rotator
    if args.lowpass_cutoff is not None:
        transition = max(args.lowpass_cutoff * 0.2, args.input_rate / 10_000.0)
        taps = firdes.low_pass(1.0, args.input_rate, args.lowpass_cutoff, transition)
        lowpass = filter.fir_filter_ccf(1, taps)
        flowgraph.connect(current, lowpass)
        current = lowpass
    if not math.isclose(args.input_rate, args.output_rate, rel_tol=0.0, abs_tol=1e-12):
        ratio = Fraction(args.output_rate / args.input_rate).limit_denominator(1_000_000)
        resampler = filter.rational_resampler_ccc(interpolation=ratio.numerator, decimation=ratio.denominator)
        flowgraph.connect(current, resampler)
        current = resampler
    sink = blocks.file_sink(gr.sizeof_gr_complex, args.output, False)
    _tune_buffer(sink, output=False)
    flowgraph.connect(current, sink)
    flowgraph.run()


if __name__ == "__main__":
    main()
