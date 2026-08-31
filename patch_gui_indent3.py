with open('signal_analysis/gui.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("if HAS_QT:\nif HAS_QT:\n    class CollapsibleSection", "if HAS_QT:\n    class CollapsibleSection")

with open('signal_analysis/gui.py', 'w', encoding='utf-8') as f:
    f.write(content)
