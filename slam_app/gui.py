# pyright: reportMissingImports=false

import argparse
import copy
import traceback

import yaml
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .config import build_main_parser, save_runs_config, validate_runs_config_data
from .pipeline import run_tracking


class BatchRunner(QObject):
    log = Signal(str)
    run_status = Signal(int, str)
    done = Signal(int, int, int, int)
    failed = Signal(str)

    def __init__(self, base_args, defaults, runs, selected_indices, continue_on_error):
        super().__init__()
        self.base_args = base_args
        self.defaults = defaults
        self.runs = runs
        self.selected_indices = selected_indices
        self.continue_on_error = continue_on_error

    def run(self):
        succeeded = 0
        failed = 0
        skipped = 0

        for idx in self.selected_indices:
            run = self.runs[idx]
            run_name = run.get("name", f"run_{idx + 1:02d}")

            if bool(run.get("skip", False)):
                skipped += 1
                self.run_status.emit(idx, "skipped")
                self.log.emit(f"⏭️  Skipped run: {run_name} (skip=true)")
                continue

            run_args_dict = copy.deepcopy(vars(self.base_args))
            run_args_dict.update(self.defaults)
            run_args_dict.update(run)
            run_args = argparse.Namespace(**run_args_dict)

            self.run_status.emit(idx, "running")
            self.log.emit(f"▶️  Starting run: {run_name}")

            try:
                run_tracking(run_args, run_name=run_name)
                succeeded += 1
                self.run_status.emit(idx, "done")
                self.log.emit(f"✅ Completed run: {run_name}")
            except Exception as exc:
                failed += 1
                self.run_status.emit(idx, "failed")
                self.log.emit(f"❌ Failed run: {run_name} ({exc})")
                if not self.continue_on_error:
                    break

        self.done.emit(len(self.selected_indices), succeeded, skipped, failed)


class MainWindow(QMainWindow):
    STATUS_PREFIX = {
        "idle": "[idle] ",
        "running": "[running] ",
        "done": "[done] ",
        "failed": "[failed] ",
        "skipped": "[skipped] ",
    }

    def __init__(self, config_path):
        super().__init__()
        self.setWindowTitle("DROID-SLAM Runner")
        self.resize(1300, 820)

        self.config_path = config_path
        self.config_data = {"defaults": {}, "runs": []}
        self.status_by_index = {}
        self.worker_thread = None

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        top.addWidget(QLabel("Runs config:"))
        self.path_edit = QLineEdit(self.config_path)
        top.addWidget(self.path_edit, 1)
        self.btn_browse = QPushButton("Browse")
        self.btn_load = QPushButton("Load")
        self.btn_save = QPushButton("Save")
        top.addWidget(self.btn_browse)
        top.addWidget(self.btn_load)
        top.addWidget(self.btn_save)
        root.addLayout(top)

        actions = QHBoxLayout()
        self.btn_validate = QPushButton("Validate YAML")
        self.btn_add_run = QPushButton("Add Run Template")
        self.btn_remove_run = QPushButton("Remove Selected Run")
        self.continue_on_error = QCheckBox("Continue on error")
        self.continue_on_error.setChecked(True)
        self.btn_run = QPushButton("Run Selected")
        actions.addWidget(self.btn_validate)
        actions.addWidget(self.btn_add_run)
        actions.addWidget(self.btn_remove_run)
        actions.addWidget(self.continue_on_error)
        actions.addStretch(1)
        actions.addWidget(self.btn_run)
        root.addLayout(actions)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Runs (check to execute):"))
        self.run_list = QListWidget()
        left_layout.addWidget(self.run_list, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("YAML editor:"))
        self.yaml_edit = QPlainTextEdit()
        right_layout.addWidget(self.yaml_edit, 2)
        right_layout.addWidget(QLabel("Execution log:"))
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        right_layout.addWidget(self.log_edit, 1)
        splitter.addWidget(right)

        splitter.setSizes([380, 920])

        self.btn_browse.clicked.connect(self.on_browse)
        self.btn_load.clicked.connect(self.on_load)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_validate.clicked.connect(self.on_validate)
        self.btn_add_run.clicked.connect(self.on_add_run)
        self.btn_remove_run.clicked.connect(self.on_remove_run)
        self.btn_run.clicked.connect(self.on_run_selected)

        self.on_load()

    def _append_log(self, message):
        self.log_edit.appendPlainText(message)

    def _run_title(self, idx, run):
        name = run.get("name", f"run_{idx + 1:02d}")
        status = self.status_by_index.get(idx, "idle")
        return f"{self.STATUS_PREFIX[status]}{name}"

    def _refresh_run_list(self):
        self.run_list.clear()
        runs = self.config_data.get("runs", [])
        for idx, run in enumerate(runs):
            item = QListWidgetItem(self._run_title(idx, run))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked if bool(run.get("skip", False)) else Qt.Checked)
            self.run_list.addItem(item)

    def _parse_editor(self):
        text = self.yaml_edit.toPlainText()
        config = yaml.safe_load(text) or {}
        defaults, runs = validate_runs_config_data(config)
        return config, defaults, runs

    def on_browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open runs config", self.path_edit.text(), "YAML Files (*.yaml *.yml)")
        if path:
            self.path_edit.setText(path)

    def on_load(self):
        self.config_path = self.path_edit.text().strip()
        if not self.config_path:
            QMessageBox.warning(self, "Missing path", "Please provide a runs config path.")
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config_data = yaml.safe_load(f) or {"defaults": {}, "runs": []}
            self.config_data.setdefault("defaults", {})
            self.config_data.setdefault("runs", [])
            self.status_by_index = {i: "idle" for i in range(len(self.config_data["runs"]))}
            self.yaml_edit.setPlainText(yaml.dump(self.config_data, default_flow_style=False, sort_keys=False, allow_unicode=True))
            self._refresh_run_list()
            self._append_log(f"📚 Loaded: {self.config_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))

    def on_save(self):
        self.config_path = self.path_edit.text().strip()
        try:
            config, _, _ = self._parse_editor()
            save_runs_config(self.config_path, config)
            self.config_data = config
            self.status_by_index = {i: self.status_by_index.get(i, "idle") for i in range(len(self.config_data["runs"]))}
            self._refresh_run_list()
            self._append_log(f"💾 Saved: {self.config_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def on_validate(self):
        try:
            config, _, _ = self._parse_editor()
            self.config_data = config
            self.status_by_index = {i: self.status_by_index.get(i, "idle") for i in range(len(self.config_data["runs"]))}
            self._refresh_run_list()
            QMessageBox.information(self, "Valid", "YAML config is valid.")
        except Exception as exc:
            QMessageBox.critical(self, "Validation failed", str(exc))

    def on_add_run(self):
        try:
            config, _, runs = self._parse_editor()
            runs.append(
                {
                    "name": f"run_{len(runs) + 1:02d}",
                    "root_folder": "",
                    "input_folder": "",
                    "output_folder": "droid-slam",
                    "calib": "",
                    "camera_model": "radtan",
                    "skip": False,
                }
            )
            config["runs"] = runs
            self.yaml_edit.setPlainText(yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True))
            self.config_data = config
            self.status_by_index = {i: self.status_by_index.get(i, "idle") for i in range(len(runs))}
            self._refresh_run_list()
        except Exception as exc:
            QMessageBox.critical(self, "Add run failed", str(exc))

    def on_remove_run(self):
        row = self.run_list.currentRow()
        if row < 0:
            QMessageBox.information(self, "No selection", "Select a run in the list to remove it.")
            return
        try:
            config, _, runs = self._parse_editor()
            if row >= len(runs):
                raise ValueError("selected row is out of range")
            removed_name = runs[row].get("name", f"run_{row + 1:02d}")
            runs.pop(row)
            config["runs"] = runs
            self.yaml_edit.setPlainText(yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True))
            self.config_data = config
            self.status_by_index = {i: "idle" for i in range(len(runs))}
            self._refresh_run_list()
            self._append_log(f"🗑️  Removed run: {removed_name}")
        except Exception as exc:
            QMessageBox.critical(self, "Remove run failed", str(exc))

    def _set_running_controls(self, running):
        self.btn_run.setEnabled(not running)
        self.btn_load.setEnabled(not running)
        self.btn_save.setEnabled(not running)
        self.btn_validate.setEnabled(not running)

    def on_run_selected(self):
        try:
            config, defaults, runs = self._parse_editor()
            self.config_data = config
            selected_indices = []
            for i in range(self.run_list.count()):
                if self.run_list.item(i).checkState() == Qt.Checked:
                    selected_indices.append(i)
            if not selected_indices:
                QMessageBox.information(self, "No runs selected", "Check one or more runs to execute.")
                return

            base_args = build_main_parser().parse_args([])
            for idx in selected_indices:
                self.status_by_index[idx] = "idle"
            self._refresh_run_list()

            self.worker_thread = QThread(self)
            self.worker = BatchRunner(
                base_args=base_args,
                defaults=defaults,
                runs=runs,
                selected_indices=selected_indices,
                continue_on_error=self.continue_on_error.isChecked(),
            )
            self.worker.moveToThread(self.worker_thread)
            self.worker_thread.started.connect(self.worker.run)
            self.worker.log.connect(self._append_log)
            self.worker.run_status.connect(self.on_worker_run_status)
            self.worker.done.connect(self.on_worker_done)
            self.worker.failed.connect(self.on_worker_failed)
            self.worker.done.connect(self.worker_thread.quit)
            self.worker.done.connect(self.worker.deleteLater)
            self.worker_thread.finished.connect(self.worker_thread.deleteLater)

            self._set_running_controls(True)
            self.worker_thread.start()
        except Exception as exc:
            QMessageBox.critical(self, "Run start failed", str(exc))

    def on_worker_run_status(self, idx, status):
        self.status_by_index[idx] = status
        self._refresh_run_list()

    def on_worker_done(self, total, succeeded, skipped, failed):
        self._set_running_controls(False)
        self._append_log(
            f"📊 GUI batch summary: {succeeded}/{total} succeeded, {skipped} skipped, {failed} failed"
        )

    def on_worker_failed(self, err):
        self._set_running_controls(False)
        QMessageBox.critical(self, "Run failed", err)


def launch_gui(config_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(config_path)
    window.show()
    try:
        return app.exec()
    except Exception:
        traceback.print_exc()
        return 1
