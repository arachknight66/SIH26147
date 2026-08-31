with open('signal_analysis/gui.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("if HAS_QT:\ndef _format_hz", "def _format_hz")
content = content.replace("    class CollapsibleSection(QWidget):", "if HAS_QT:\n    class CollapsibleSection(QWidget):")

with open('signal_analysis/gui.py', 'w', encoding='utf-8') as f:
    f.write(content)
