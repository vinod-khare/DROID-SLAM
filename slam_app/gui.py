# pyright: reportMissingImports=false

import argparse
import copy
import traceback

import yaml
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
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
    QSpinBox,
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
        self.run_checked = {}
        self._updating_run_list = False
        self._updating_form = False
        self._updating_yaml = False
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

        details_box = QGroupBox("Run details")
        details_form = QFormLayout(details_box)

        self.field_name = QLineEdit()
        self.field_root = QLineEdit()
        self.field_input = QLineEdit()
        self.field_output = QLineEdit()
        self.field_calib = QLineEdit()
        self.field_camera = QComboBox()
        self.field_camera.addItems(["radtan", "fisheye"])
        self.field_skip = QCheckBox("Skip this run")

        self.field_stride = QSpinBox()
        self.field_stride.setRange(1, 100000)
        self.field_t0 = QSpinBox()
        self.field_t0.setRange(0, 100000000)
        self.field_buffer = QSpinBox()
        self.field_buffer.setRange(1, 100000)

        self.field_filter_thresh = QDoubleSpinBox()
        self.field_filter_thresh.setRange(0.0, 1_000_000.0)
        self.field_filter_thresh.setDecimals(4)
        self.field_keyframe_thresh = QDoubleSpinBox()
        self.field_keyframe_thresh.setRange(0.0, 1_000_000.0)
        self.field_keyframe_thresh.setDecimals(4)

        self.field_target_width = QLineEdit()
        self.field_target_width.setPlaceholderText("empty = use native width")
        self.field_target_height = QLineEdit()
        self.field_target_height.setPlaceholderText("empty = use native height")

        details_form.addRow("name", self.field_name)
        details_form.addRow("root_folder", self.field_root)
        details_form.addRow("input_folder", self.field_input)
        details_form.addRow("output_folder", self.field_output)
        details_form.addRow("calib", self.field_calib)
        details_form.addRow("camera_model", self.field_camera)
        details_form.addRow("", self.field_skip)
        details_form.addRow("stride", self.field_stride)
        details_form.addRow("t0", self.field_t0)
        details_form.addRow("buffer", self.field_buffer)
        details_form.addRow("filter_thresh", self.field_filter_thresh)
        details_form.addRow("keyframe_thresh", self.field_keyframe_thresh)
        details_form.addRow("target_width", self.field_target_width)
        details_form.addRow("target_height", self.field_target_height)

        self.btn_apply_form = QPushButton("Apply form to selected run")
        details_form.addRow("", self.btn_apply_form)
        left_layout.addWidget(details_box)

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
        self.btn_apply_form.clicked.connect(self.on_apply_form_to_run)
        self.run_list.currentRowChanged.connect(self.on_run_selection_changed)
        self.run_list.itemChanged.connect(self.on_run_item_changed)
        self.yaml_edit.textChanged.connect(self.on_yaml_text_changed)

        self.field_name.editingFinished.connect(self.on_form_edited)
        self.field_root.editingFinished.connect(self.on_form_edited)
        self.field_input.editingFinished.connect(self.on_form_edited)
        self.field_output.editingFinished.connect(self.on_form_edited)
        self.field_calib.editingFinished.connect(self.on_form_edited)
        self.field_camera.currentTextChanged.connect(self.on_form_edited)
        self.field_skip.stateChanged.connect(self.on_form_edited)
        self.field_stride.valueChanged.connect(self.on_form_edited)
        self.field_t0.valueChanged.connect(self.on_form_edited)
        self.field_buffer.valueChanged.connect(self.on_form_edited)
        self.field_filter_thresh.valueChanged.connect(self.on_form_edited)
        self.field_keyframe_thresh.valueChanged.connect(self.on_form_edited)
        self.field_target_width.editingFinished.connect(self.on_form_edited)
        self.field_target_height.editingFinished.connect(self.on_form_edited)

        self.on_load()

    def _append_log(self, message):
        self.log_edit.appendPlainText(message)

    def _set_editor_from_config(self):
        self._updating_yaml = True
        try:
            self.yaml_edit.setPlainText(
                yaml.dump(self.config_data, default_flow_style=False, sort_keys=False, allow_unicode=True)
            )
        finally:
            self._updating_yaml = False

    def _selected_run_index(self):
        row = self.run_list.currentRow()
        runs = self.config_data.get("runs", [])
        if row < 0 or row >= len(runs):
            return None
        return row

    def _run_title(self, idx, run):
        name = run.get("name", f"run_{idx + 1:02d}")
        status = self.status_by_index.get(idx, "idle")
        return f"{self.STATUS_PREFIX[status]}{name}"

    def _refresh_run_list(self):
        current_row = self.run_list.currentRow()
        self._updating_run_list = True
        self.run_list.clear()
        runs = self.config_data.get("runs", [])
        for idx, run in enumerate(runs):
            item = QListWidgetItem(self._run_title(idx, run))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            default_checked = not bool(run.get("skip", False))
            is_checked = self.run_checked.get(idx, default_checked)
            item.setCheckState(Qt.Checked if is_checked else Qt.Unchecked)
            self.run_list.addItem(item)
        self._updating_run_list = False

        if runs:
            if current_row < 0 or current_row >= len(runs):
                current_row = 0
            self.run_list.setCurrentRow(current_row)
            self._load_run_to_form(current_row)
        else:
            self._load_run_to_form(None)

    def _load_run_to_form(self, idx):
        self._updating_form = True
        try:
            if idx is None:
                self.field_name.setText("")
                self.field_root.setText("")
                self.field_input.setText("")
                self.field_output.setText("")
                self.field_calib.setText("")
                self.field_camera.setCurrentText("radtan")
                self.field_skip.setChecked(False)
                self.field_stride.setValue(1)
                self.field_t0.setValue(0)
                self.field_buffer.setValue(512)
                self.field_filter_thresh.setValue(2.4)
                self.field_keyframe_thresh.setValue(2.0)
                self.field_target_width.setText("")
                self.field_target_height.setText("")
                return

            run = self.config_data["runs"][idx]
            self.field_name.setText(str(run.get("name", f"run_{idx + 1:02d}")))
            self.field_root.setText(str(run.get("root_folder", "")))
            self.field_input.setText(str(run.get("input_folder", "")))
            self.field_output.setText(str(run.get("output_folder", "")))
            self.field_calib.setText(str(run.get("calib", "")))
            self.field_camera.setCurrentText(str(run.get("camera_model", "radtan")))
            self.field_skip.setChecked(bool(run.get("skip", False)))
            self.field_stride.setValue(int(run.get("stride", self.config_data.get("defaults", {}).get("stride", 1))))
            self.field_t0.setValue(int(run.get("t0", self.config_data.get("defaults", {}).get("t0", 0))))
            self.field_buffer.setValue(int(run.get("buffer", self.config_data.get("defaults", {}).get("buffer", 512))))
            self.field_filter_thresh.setValue(float(run.get("filter_thresh", self.config_data.get("defaults", {}).get("filter_thresh", 2.4))))
            self.field_keyframe_thresh.setValue(float(run.get("keyframe_thresh", self.config_data.get("defaults", {}).get("keyframe_thresh", 2.0))))
            self.field_target_width.setText("" if run.get("target_width", None) is None else str(run.get("target_width")))
            self.field_target_height.setText("" if run.get("target_height", None) is None else str(run.get("target_height")))
        finally:
            self._updating_form = False

    def _parse_optional_int(self, value_text, field_name):
        text = value_text.strip()
        if text == "":
            return None
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer or empty") from exc
        if value <= 0:
            raise ValueError(f"{field_name} must be positive")
        return value

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
            self.run_checked = {i: (not bool(run.get("skip", False))) for i, run in enumerate(self.config_data["runs"])}
            self._set_editor_from_config()
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
            self.run_checked = {i: self.run_checked.get(i, not bool(run.get("skip", False))) for i, run in enumerate(self.config_data["runs"])}
            self._refresh_run_list()
            self._append_log(f"💾 Saved: {self.config_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def on_validate(self):
        try:
            config, _, _ = self._parse_editor()
            self.config_data = config
            self.status_by_index = {i: self.status_by_index.get(i, "idle") for i in range(len(self.config_data["runs"]))}
            self.run_checked = {i: self.run_checked.get(i, not bool(run.get("skip", False))) for i, run in enumerate(self.config_data["runs"])}
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
            self.config_data = config
            self._set_editor_from_config()
            self.status_by_index = {i: self.status_by_index.get(i, "idle") for i in range(len(runs))}
            self.run_checked = {i: self.run_checked.get(i, not bool(run.get("skip", False))) for i, run in enumerate(runs)}
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
            self.config_data = config
            self._set_editor_from_config()
            self.status_by_index = {i: "idle" for i in range(len(runs))}
            self.run_checked = {i: (not bool(run.get("skip", False))) for i, run in enumerate(runs)}
            self._refresh_run_list()
            self._append_log(f"🗑️  Removed run: {removed_name}")
        except Exception as exc:
            QMessageBox.critical(self, "Remove run failed", str(exc))

    def on_run_selection_changed(self, row):
        if row < 0:
            self._load_run_to_form(None)
        else:
            self._load_run_to_form(row)

    def on_run_item_changed(self, item):
        if self._updating_run_list:
            return
        idx = self.run_list.row(item)
        if idx < 0:
            return
        is_checked = item.checkState() == Qt.Checked
        self.run_checked[idx] = is_checked

        runs = self.config_data.get("runs", [])
        if idx < len(runs):
            # List checkbox means "run enabled"; skip is the inverse.
            runs[idx]["skip"] = not is_checked
            self._set_editor_from_config()
            self._refresh_run_list()
            self.run_list.setCurrentRow(idx)

    def _apply_form_to_selected_run(self, log_update=True):
        if self._updating_form:
            return
        idx = self._selected_run_index()
        if idx is None:
            return

        run = self.config_data["runs"][idx]
        run["name"] = self.field_name.text().strip() or f"run_{idx + 1:02d}"
        run["root_folder"] = self.field_root.text().strip()
        run["input_folder"] = self.field_input.text().strip()
        run["output_folder"] = self.field_output.text().strip()
        run["calib"] = self.field_calib.text().strip()
        run["camera_model"] = self.field_camera.currentText()
        run["skip"] = self.field_skip.isChecked()
        run["stride"] = int(self.field_stride.value())
        run["t0"] = int(self.field_t0.value())
        run["buffer"] = int(self.field_buffer.value())
        run["filter_thresh"] = float(self.field_filter_thresh.value())
        run["keyframe_thresh"] = float(self.field_keyframe_thresh.value())

        target_width = self._parse_optional_int(self.field_target_width.text(), "target_width")
        target_height = self._parse_optional_int(self.field_target_height.text(), "target_height")
        if (target_width is None) != (target_height is None):
            raise ValueError("target_width and target_height must both be set or both be empty")
        if target_width is None:
            run.pop("target_width", None)
            run.pop("target_height", None)
        else:
            run["target_width"] = target_width
            run["target_height"] = target_height

        self.run_checked[idx] = not run["skip"]
        self._set_editor_from_config()
        self._refresh_run_list()
        self.run_list.setCurrentRow(idx)
        if log_update:
            self._append_log(f"🛠️  Updated run: {run['name']}")

    def on_form_edited(self):
        if self._updating_form:
            return
        try:
            self._apply_form_to_selected_run(log_update=False)
        except Exception:
            # Keep UI responsive while typing partial values; explicit Apply will report details.
            pass

    def on_yaml_text_changed(self):
        if self._updating_yaml:
            return
        try:
            config, _, _ = self._parse_editor()
        except Exception:
            # Ignore transient invalid YAML while user is typing.
            return

        current_row = self.run_list.currentRow()
        self.config_data = config
        self.status_by_index = {
            i: self.status_by_index.get(i, "idle") for i in range(len(self.config_data.get("runs", [])))
        }
        self.run_checked = {
            i: self.run_checked.get(i, not bool(run.get("skip", False)))
            for i, run in enumerate(self.config_data.get("runs", []))
        }
        self._refresh_run_list()
        if 0 <= current_row < self.run_list.count():
            self.run_list.setCurrentRow(current_row)

    def on_apply_form_to_run(self):
        if self._updating_form:
            return
        idx = self._selected_run_index()
        if idx is None:
            QMessageBox.information(self, "No run selected", "Select a run first.")
            return

        try:
            self._apply_form_to_selected_run(log_update=True)
        except Exception as exc:
            QMessageBox.critical(self, "Apply failed", str(exc))

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
