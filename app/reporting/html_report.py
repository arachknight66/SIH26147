from __future__ import annotations
from app.orchestration.pipeline_runner import PipelineResult

def build_html_report(result: PipelineResult) -> str:
    """
    Generate a standalone, interactive, publication-ready HTML report
    with clean typography and CSS styling.
    """
    p2 = result.phase2_result.output if (result.phase2_result and result.phase2_result.output) else None
    p3 = result.phase3_result.output if (result.phase3_result and result.phase3_result.output) else None
    p4 = result.phase4_result.output if (result.phase4_result and result.phase4_result.output) else None
    p5 = result.phase5_result.output if (result.phase5_result and result.phase5_result.output) else None
    p6 = result.phase6_result.output if (result.phase6_result and result.phase6_result.output) else None
    prov = result.provenance

    status_color = "#10b981" if result.is_verified else "#ef4444" if (p6 and p6.is_falsified) else "#f59e0b"
    status_text = p6.status.value.upper() if p6 else ("FAILED" if result.failure else "UNKNOWN")

    claims_rows = ""
    if p6 and p6.claims:
        for c in p6.claims:
            c_color = "#10b981" if c.status.value == "supported" else "#ef4444" if c.status.value == "falsified" else "#f59e0b"
            claims_rows += f"""
            <tr>
                <td><b>Claim {c.claim_id}</b></td>
                <td>{c.claim_text}</td>
                <td><span style="color: {c_color}; font-weight: bold;">{c.status.value.upper()}</span></td>
                <td>{c.confidence:.2f}</td>
                <td>{c.independence_level.value}</td>
            </tr>
            """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SIH26147 Scientific Verification Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 32px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
        h1, h2, h3 {{ color: #38bdf8; margin-top: 0; }}
        .badge {{ display: inline-block; padding: 6px 12px; border-radius: 6px; font-weight: bold; background: {status_color}; color: #0f172a; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .card {{ background: #334155; padding: 16px; border-radius: 8px; }}
        .card-label {{ font-size: 0.85em; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px; }}
        .card-val {{ font-size: 1.25em; font-weight: 600; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; margin-bottom: 24px; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #475569; }}
        th {{ background: #334155; color: #38bdf8; }}
        .hash {{ font-family: monospace; background: #0f172a; padding: 8px; border-radius: 4px; word-break: break-all; }}
        .footer {{ font-size: 0.85em; color: #64748b; margin-top: 32px; text-align: center; }}
    </style>
</head>
<body>
<div class="container">
    <h1>SIH26147 Scientific Signal Recovery & Verification Report</h1>
    <div style="margin-bottom: 24px;">
        <span class="badge">{status_text}</span>
        <span style="margin-left: 12px; color: #94a3b8;">Source: {result.input_path or 'In-Memory Stream'}</span>
    </div>

    <h2>Executive Summary</h2>
    <div class="grid">
        <div class="card">
            <div class="card-label">Modulation</div>
            <div class="card-val">{p3.selected_hypothesis.label if (p3 and p3.selected_hypothesis) else 'UNKNOWN'}</div>
        </div>
        <div class="card">
            <div class="card-label">Symbol Rate (Normalized)</div>
            <div class="card-val">{p4.recovered_signal.samples_per_symbol if (p4 and p4.recovered_signal) else 'UNKNOWN'} SPS</div>
        </div>
        <div class="card">
            <div class="card-label">FEC Code</div>
            <div class="card-val">{p5.selected_candidate.fec.code_name if (p5 and p5.selected_candidate and p5.selected_candidate.fec) else 'NONE'}</div>
        </div>
        <div class="card">
            <div class="card-label">Integrity (CRC)</div>
            <div class="card-val">{p5.selected_candidate.integrity.crc_results[0].crc_name if (p5 and p5.selected_candidate and p5.selected_candidate.integrity and p5.selected_candidate.integrity.crc_results) else 'NONE'}</div>
        </div>
    </div>

    <h2>Independent 7-Claim Verification Matrix</h2>
    <table>
        <thead>
            <tr>
                <th>Claim ID</th>
                <th>Claim Description</th>
                <th>Audit Status</th>
                <th>Confidence</th>
                <th>Independence</th>
            </tr>
        </thead>
        <tbody>
            {claims_rows}
        </tbody>
    </table>

    <h2>Error Budget & Uncertainty</h2>
    <div class="grid">
        <div class="card">
            <div class="card-label">Carrier Uncertainty</div>
            <div class="card-val">{p6.error_budget.carrier_uncertainty if (p6 and p6.error_budget) else 0.0:.4f}</div>
        </div>
        <div class="card">
            <div class="card-label">Timing Uncertainty</div>
            <div class="card-val">{p6.error_budget.timing_uncertainty if (p6 and p6.error_budget) else 0.0:.4f}</div>
        </div>
        <div class="card">
            <div class="card-label">BER Proxy</div>
            <div class="card-val">{p6.error_budget.bit_error_rate_proxy if (p6 and p6.error_budget) else 0.0:.4f}</div>
        </div>
        <div class="card">
            <div class="card-label">Composite Uncertainty</div>
            <div class="card-val">{p6.error_budget.total_composite_uncertainty if (p6 and p6.error_budget) else 0.0:.4f}</div>
        </div>
    </div>

    <h2>Reproducibility & Provenance</h2>
    <p><b>Reproducibility SHA-256 Hash:</b></p>
    <div class="hash">{prov.reproducibility_hash if prov else 'N/A'}</div>
    <p><b>Input Data SHA-256:</b></p>
    <div class="hash">{result.input_sha256}</div>

    <div class="footer">
        Generated by SIH26147 v0.7.0 • Total Execution Time: {result.total_duration_seconds:.2f}s
    </div>
</div>
</body>
</html>
"""

def export_html_report(result: PipelineResult, file_path: str) -> None:
    html_content = build_html_report(result)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
