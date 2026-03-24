import os
from pathlib import Path

import numpy as np
import pytest
from PyQt6 import QtWidgets
from PyQt6.QtCore import QSettings

from simView import GUIapp

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture
def bruker_fixture_path() -> Path:
    return Path(__file__).resolve().parents[1] / "testData" / "mrScanSim"


@pytest.fixture
def isolated_settings(tmp_path: Path) -> None:
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    settings = QSettings("MR_ISIBrno", "BrukerSimView")
    settings.clear()
    settings.sync()


def test_jump_targets_available_for_bruker_fixture(
    qapp: QtWidgets.QApplication,
    isolated_settings: None,
    bruker_fixture_path: Path,
) -> None:
    gui = GUIapp(simPath=str(bruker_fixture_path))
    try:
        targets = gui.get_pulse_program_jump_targets()
        assert len(targets) > 0
        assert any("ln " in target[0] for target in targets)
    finally:
        gui.close()
        gui.deleteLater()


def test_jump_to_ppg_line_action_updates_view_to_selected_target(
    qapp: QtWidgets.QApplication,
    isolated_settings: None,
    bruker_fixture_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gui = GUIapp(simPath=str(bruker_fixture_path))
    try:
        targets = gui.get_pulse_program_jump_targets()
        assert len(targets) > 0
        target_index = len(targets) - 1
        selected_label = targets[target_index][0]
        target_time = targets[target_index][1]
        gui.windowWidth = max(gui.sliderScaler * 100, 1e-3)
        gui.updateView()

        monkeypatch.setattr(
            QtWidgets.QInputDialog,
            "getItem",
            staticmethod(lambda *args, **kwargs: (selected_label, True)),
        )

        gui.jump_to_ppg_line()

        assert np.isclose(float(gui.currentCursorTime), target_time)
        x_min, x_max = gui.plots[0].viewRange()[0]
        assert x_min <= target_time <= x_max
    finally:
        gui.close()
        gui.deleteLater()


def test_channel_checkbox_toggle_hides_plot_container(
    qapp: QtWidgets.QApplication,
    isolated_settings: None,
) -> None:
    data = {
        "time": {"val": [0.0, 1.0, 2.0], "units": "ms", "show": "yes"},
        "sig_a": {"val": [0.0, 1.0, 0.0], "units": "a.u.", "show": "yes"},
        "sig_b": {"val": [1.0, 0.0, 1.0], "units": "a.u.", "show": "yes"},
    }
    gui = GUIapp(data=data)
    try:
        assert len(gui.checkBoxes) >= 2
        first_check_box = gui.checkBoxes[0]
        first_container = gui.plotContainers[first_check_box.contID]
        assert not first_container.isHidden()

        first_check_box.setChecked(False)
        qapp.processEvents()
        assert first_container.isHidden()

        first_check_box.setChecked(True)
        qapp.processEvents()
        assert not first_container.isHidden()
    finally:
        gui.close()
        gui.deleteLater()


def test_jump_to_next_and_previous_rf_pulse_updates_cursor_time(
    qapp: QtWidgets.QApplication,
    isolated_settings: None,
) -> None:
    data = {
        "time": {"val": [0.0, 1.0, 2.0, 3.0, 4.0], "units": "ms", "show": "yes"},
        "rf_am": {"val": [0.0, 10.0, 10.0, 0.0, 20.0], "units": "%", "show": "yes"},
    }
    gui = GUIapp(data=data)
    try:
        assert gui.nextRfPulseButton.isEnabled()
        assert gui.prevRfPulseButton.isEnabled()

        gui.update_status(cursor_time=0.0)
        gui.jump_to_next_rf_pulse()
        assert np.isclose(float(gui.currentCursorTime), 0.001)

        gui.jump_to_next_rf_pulse()
        assert np.isclose(float(gui.currentCursorTime), 0.004)

        gui.jump_to_previous_rf_pulse()
        assert np.isclose(float(gui.currentCursorTime), 0.001)
    finally:
        gui.close()
        gui.deleteLater()
