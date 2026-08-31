import re
with open('signal_analysis/cli.py', 'r') as f:
    content = f.read()

content = content.replace(
    '    parser.add_argument("--output", choices=["json", "text"], default="json", help="Output format")',
    '    parser.add_argument("--output", choices=["json", "text"], default="json", help="Output format")\n    parser.add_argument("--wav-stereo-mode", choices=["unresolved", "stereo_real", "stereo_iq"], default="unresolved", help="Stereo interpretation for WAV files")'
)

content = content.replace(
    "                reader = WavReader(path_str, mode='unresolved')",
    "                reader = WavReader(path_str, mode=args.wav_stereo_mode)"
)

with open('signal_analysis/cli.py', 'w') as f:
    f.write(content)
