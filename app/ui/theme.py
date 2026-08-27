from __future__ import annotations

# Scientific Dark Theme Palette
DARK_THEME = """
QMainWindow, QWidget {
    background-color: #0f172a;
    color: #f8fafc;
    font-family: "Segoe UI", "SF Pro Display", -apple-system, sans-serif;
    font-size: 13px;
}

QListWidget {
    background-color: #1e293b;
    border: none;
    border-radius: 8px;
    padding: 8px;
    color: #94a3b8;
    font-weight: 500;
}

QListWidget::item {
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 4px;
}

QListWidget::item:selected {
    background-color: #38bdf8;
    color: #0f172a;
    font-weight: bold;
}

QListWidget::item:hover:!selected {
    background-color: #334155;
    color: #f8fafc;
}

QTableWidget {
    background-color: #1e293b;
    gridline-color: #334155;
    border: 1px solid #334155;
    border-radius: 8px;
    color: #f8fafc;
}

QHeaderView::section {
    background-color: #334155;
    color: #38bdf8;
    padding: 8px;
    font-weight: bold;
    border: 1px solid #1e293b;
}

QProgressBar {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    text-align: center;
    color: #f8fafc;
    font-weight: bold;
}

QProgressBar::chunk {
    background-color: #38bdf8;
    border-radius: 5px;
}

QPushButton {
    background-color: #38bdf8;
    color: #0f172a;
    font-weight: bold;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
}

QPushButton:hover {
    background-color: #7dd3fc;
}

QPushButton:pressed {
    background-color: #0284c7;
}

QPushButton#secondaryBtn {
    background-color: #334155;
    color: #f8fafc;
    border: 1px solid #475569;
}

QPushButton#secondaryBtn:hover {
    background-color: #475569;
}

QGroupBox {
    border: 1px solid #334155;
    border-radius: 8px;
    margin-top: 12px;
    font-weight: bold;
    color: #38bdf8;
    padding: 16px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}

QTextEdit, QPlainTextEdit, QLineEdit {
    background-color: #1e293b;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px;
}

QStatusBar {
    background-color: #1e293b;
    color: #94a3b8;
    border-top: 1px solid #334155;
}
"""

EPISTEMIC_COLORS = {
    "OBSERVED": "#38bdf8",              # Sky blue
    "INFERRED": "#a855f7",              # Purple
    "ASSUMED": "#f59e0b",               # Amber
    "CORRECTED": "#06b6d4",             # Cyan
    "SUPPORTED": "#10b981",             # Emerald green
    "INDEPENDENTLY_VERIFIED": "#10b981",# Green
    "AMBIGUOUS": "#eab308",             # Yellow
    "REJECTED": "#ef4444",              # Red
    "FALSIFIED": "#ef4444",             # Red
    "UNKNOWN": "#64748b",               # Slate gray
}
