import os

with open('signal_analysis/gui.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.endswith('\n"'):
        # Fix the unescaped newline within string
        line = line.replace('\n"', '\\n"')
    
    # Check for specific split strings
    if line.strip() == '"':
        # the previous line had an unterminated string
        new_lines[-1] = new_lines[-1].rstrip() + '\\n"\n'
        continue
        
    if 'details += f"Top Hyp: {h.label} [{h.status.value}] (Score {h.score:.2f})' in line:
        line = line.rstrip() + '\\n"\n'
    elif 'details += f"Sync EVM: {s.evm_percent:.1f}% ' in line:
        line = line.rstrip() + '\\n" + \\\n'
    elif 'CFO: {s.cfo_estimate:.2f} {s.cfo_unit}' in line:
        line = ' ' * 12 + f'f"CFO: {{s.cfo_estimate:.2f}} {{s.cfo_unit}}\\n"\n'
    elif 'details += f"FEC: {r.codec_name} -> {r.decode_success}' in line:
        line = line.rstrip() + '\\n" + \\\n'
    elif 'Corrected: {r.corrected_bit_count} bits' in line:
        line = ' ' * 12 + f'f"Corrected: {{r.corrected_bit_count}} bits\\n"\n'
    elif 'details += f"FEC Diag: {r.diagnostics[0].message}' in line:
        line = line.rstrip() + '\\n"\n'
    elif 'details += f"Framing Status: [{fs.status.value}]' in line:
        line = line.rstrip() + '\\n"\n'
    elif 'details += f"Sync Word: {fs.header_match.pattern.name} at {fs.header_match.bit_offset}' in line:
        line = line.rstrip() + '\\n"\n'
    elif 'details += f"Periodicity: {fs.header_match.periodicity_consistent}' in line:
        line = line.rstrip() + '\\n"\n'
    elif 'details += f"CRC: {fs.crc_candidate.polynomial_name} verified={fs.crc_candidate.verified}' in line:
        line = line.rstrip() + '\\n"\n'
        
    new_lines.append(line)

with open('signal_analysis/gui.py', 'w') as f:
    f.writelines(new_lines)
