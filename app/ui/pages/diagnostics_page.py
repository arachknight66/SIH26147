from __future__ import annotations
from typing import Any
from app.deployment.environment import get_system_environment
from app.deployment.diagnostics import run_self_diagnostics

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QPushButton, QLabel, QHBoxLayout
    from PySide6.QtGui import QColor
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

class DiagnosticsPage(QWidget):
    """
    System Environment, Dependency Status, and Health Checks Page.
    """
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        if HAS_QT:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("<h2>System Environment & Computational Health Checks</h2>"))

            btn_box = QHBoxLayout()
            self.btn_run_diag = QPushButton("Run Self-Diagnostics Suite")
            self.btn_run_diag.clicked.connect(self._run_diagnostics)
            btn_box.addWidget(self.btn_run_diag)
            btn_box.addStretch()
            layout.addLayout(btn_box)

            self.table = QTableWidget(0, 4)
            self.table.setHorizontalHeaderLabels(["Component", "Status", "Version", "Description"])
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            layout.addWidget(self.table)

            self._refresh_environment()

    def _refresh_environment(self) -> None:
        if not HAS_QT:
            return
        env = get_system_environment()
        deps = env.get("dependencies", [])
        self.table.setRowCount(len(deps))
        for r, d in enumerate(deps):
            self.table.setItem(r, 0, QTableWidgetItem(d["name"]))
            st_item = QTableWidgetItem("AVAILABLE" if d["is_installed"] else ("OPTIONAL (Missing)" if not d["is_required"] else "REQUIRED (Missing)"))
            if d["is_installed"]:
                st_item.setForeground(QColor("#10b981"))
            elif d["is_required"]:
                st_item.setForeground(QColor("#ef4444"))
            else:
                st_item.setForeground(QColor("#64748b"))

            self.table.setItem(r, 1, st_item)
            self.table.setItem(r, 2, QTableWidgetItem(str(d["version"])))
            self.table.setItem(r, 3, QTableWidgetItem(d["description"]))

    def _run_diagnostics(self) -> None:
        if not HAS_QT:
            return
        diag = run_self_diagnostics()
        # Add self-diagnostics results
        row_offset = self.table.rowCount()
        self.table.setRowCount(row_offset + len(diag))
        for i, (k, v) in enumerate(diag.items()):
            r = row_offset + i
            self.table.setItem(r, 0, QTableWidgetItem(f"Self-Test: {k}"))
            item_st = QTableWidgetItem(str(v))
            if v in ("PASS", "HEALTHY"):
                item_st.setForeground(QColor("#10b981"))
            else:
                item_st.setForeground(QColor("#ef4444"))
            self.table.setItem(r, 1, item_st)
            self.table.setItem(r, 2, QTableWidgetItem("Core"))
            self.table.setItem(r, 3, QTableWidgetItem("Automated diagnostic pipeline check"))
