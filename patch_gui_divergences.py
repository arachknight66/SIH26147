import re

with open('signal_analysis/gui.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Fix FEC panel
old_fec = r'''            if pipe_res.fec_status == PipelineStageStatus.COMPLETED:
                deint = pipe_res.deint_result
                fec = pipe_res.fec_result
                self.fec_text.setText\(
                    format_stage_status\(pipe_res.fec_status, pipe_res.sync_status\) \+ "<br>" \+
                    f"<b>Interleaver:</b> \{deint.hypothesis.family.name\}<br>"
                    f"<b>FEC Scheme:</b> \{fec.scheme_name\}<br>"
                    f"<b>FEC Corrected Bits:</b> \{fec.corrected_bit_count\} \(\{fec.corrected_bit_fraction\*100:\.1f\}%\)<br>"
                    f"<b>FEC Success:</b> \{fec.success\}"
                \)'''

new_fec = '''            if pipe_res.fec_status == PipelineStageStatus.COMPLETED:
                deint = pipe_res.deint_result
                fec = pipe_res.fec_result
                self.fec_text.setText(
                    format_stage_status(pipe_res.fec_status, pipe_res.sync_status) + "<br>" +
                    f"<b>Interleaver:</b> {deint.hypothesis.family.name}<br>"
                    f"<b>FEC Scheme:</b> {fec.codec_name}<br>"
                    f"<b>FEC Corrected Bits:</b> {fec.corrected_bit_count} ({fec.corrected_bit_fraction*100:.1f}%)<br>"
                    f"<b>FEC Success:</b> {fec.decode_success}"
                )'''
                
# Fix Framing panel
old_framing = r'''            if pipe_res.framing_status == PipelineStageStatus.COMPLETED and pipe_res.frame_structure:
                fs = pipe_res.frame_structure
                self.framing_text.setText\(
                    format_stage_status\(pipe_res.framing_status, pipe_res.fec_status\) \+ "<br>" \+
                    f"<b>Sync Word:</b> \{fs.sync_word_name\}<br>"
                    f"<b>Frame Length:</b> \{fs.frame_length_bits\} bits<br>"
                    f"<b>Payload Length:</b> \{fs.payload_length_bits\} bits<br>"
                    f"<b>CRC Type:</b> \{fs.crc_type\}<br>"
                    f"<b>Valid Frames:</b> \{fs.valid_frames_count\}/\{fs.total_frames_found\}"
                \)'''

new_framing = '''            if pipe_res.framing_status == PipelineStageStatus.COMPLETED and pipe_res.frame_structure:
                fs = pipe_res.frame_structure
                sync_name = fs.header_match.pattern.name if fs.header_match.pattern else "Unknown"
                crc_name = fs.crc_candidate.polynomial_name if fs.crc_candidate else "None"
                flen = fs.header_length_bits + (fs.payload_length_bits or 0)
                self.framing_text.setText(
                    format_stage_status(pipe_res.framing_status, pipe_res.fec_status) + "<br>" +
                    f"<b>Sync Word:</b> {sync_name}<br>"
                    f"<b>Frame Length:</b> {flen} bits<br>"
                    f"<b>Payload Length:</b> {fs.payload_length_bits} bits<br>"
                    f"<b>CRC Type:</b> {crc_name}<br>"
                    f"<b>Valid Frames:</b> N/A"
                )'''

# Fix Bitstream extract
old_bits = r'''            # Bitstream
            final_bits = None
            if pipe_res.frame_structure and len\(pipe_res.frame_structure.payloads\) > 0:
                final_bits = np.concatenate\(pipe_res.frame_structure.payloads\)
            elif pipe_res.fec_result is not None and len\(pipe_res.fec_result.decoded_bits\) > 0:
                final_bits = pipe_res.fec_result.decoded_bits
            elif pipe_res.demod_result is not None:
                final_bits = pipe_res.demod_result.hard_bits'''

new_bits = '''            # Bitstream
            final_bits = None
            if pipe_res.frame_structure and pipe_res.fec_result is not None and len(pipe_res.fec_result.decoded_bits) > 0:
                start = pipe_res.frame_structure.payload_start_bit
                end = start + (pipe_res.frame_structure.payload_length_bits or len(pipe_res.fec_result.decoded_bits))
                final_bits = pipe_res.fec_result.decoded_bits[start:end]
            elif pipe_res.fec_result is not None and len(pipe_res.fec_result.decoded_bits) > 0:
                final_bits = pipe_res.fec_result.decoded_bits
            elif pipe_res.demod_result is not None:
                final_bits = pipe_res.demod_result.hard_bits'''

code = re.sub(old_fec, new_fec, code)
code = re.sub(old_framing, new_framing, code)
code = re.sub(old_bits, new_bits, code)

with open('signal_analysis/gui.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("GUI divergences patched.")
