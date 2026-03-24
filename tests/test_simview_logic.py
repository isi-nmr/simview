from pathlib import Path

import numpy as np
import pytest

from simView import PROTON_GAMMA_MHZ_PER_T, GUIapp
from utils.simUtilsBrkr import getRFEvents, readBrkrChannels


def make_app() -> GUIapp:
    app = GUIapp.__new__(GUIapp)
    app.gradientCalibrationHzPerMm = 0.0
    app.nucleusGammaMHzPerT = PROTON_GAMMA_MHZ_PER_T
    app.displayGradientsInMtPerM = False
    app.trajectoryZeroReferenceTime = None
    app.derivedSignalStartupPadding = 1.0
    app.channels = []
    app.selectedChannels = []
    app.plots = []
    return app


class DummyProgress:
    def setLabelText(self, text: str) -> None:
        self.text = text

    def setValue(self, value: int) -> None:
        self.value = value

    def wasCanceled(self) -> bool:
        return False


class DummyMainWindow:
    def __init__(self) -> None:
        self.window_title = ""

    def setWindowTitle(self, title: str) -> None:
        self.window_title = title


class DummyPlot:
    def __init__(self) -> None:
        self.mode = None

    def set_interaction_mode(self, mode: str) -> None:
        self.mode = mode


@pytest.fixture
def app_logic() -> GUIapp:
    return make_app()


@pytest.fixture
def bruker_fixture_path() -> Path:
    return Path(__file__).resolve().parents[1] / "testData" / "mrScanSim"


def test_read_bruker_channels_from_fixture_loads_gradients_and_ncos(bruker_fixture_path: Path) -> None:
    progress = DummyProgress()
    window = DummyMainWindow()

    channels = readBrkrChannels(str(bruker_fixture_path), progress, window)

    assert channels is not None
    assert len(channels) > 1
    assert "originPPG:" in window.window_title

    gradient_channel = channels[-1]
    assert gradient_channel[0]["chanLabel"] == "Gradients"
    assert [line["key"] for line in gradient_channel] == ["Gx", "Gy", "Gz"]
    assert all(line["units"] == "%" for line in gradient_channel)
    assert all(line["raw_units"] == "%" for line in gradient_channel)
    assert all(line["t"].size == gradient_channel[0]["t"].size for line in gradient_channel)

    nco_keys = {channel[0]["key"] for channel in channels[:-1]}
    assert "am" in nco_keys
    assert "pw" in nco_keys


def test_read_bruker_channels_keeps_gradient_raw_data_copies(bruker_fixture_path: Path) -> None:
    progress = DummyProgress()
    window = DummyMainWindow()

    channels = readBrkrChannels(str(bruker_fixture_path), progress, window)

    assert channels is not None
    gradient_channel = channels[-1]
    for line in gradient_channel:
        assert "raw_data" in line
        assert not np.shares_memory(line["raw_data"], line["data"])
        np.testing.assert_allclose(line["raw_data"], line["data"])


def test_gradient_duty_cycle_uses_startup_padding(app_logic: GUIapp) -> None:
    time = np.array([0.0, 1.0, 2.0])
    data = np.array([1.0, 1.0, 0.0])

    duty_cycle = app_logic.compute_gradient_duty_cycle(time, data)

    np.testing.assert_allclose(duty_cycle, np.array([0.0, 50.0, 66.66666667]), rtol=1e-7, atol=1e-7)


def test_zero_trajectory_to_reference_interpolates_at_cursor_time(app_logic: GUIapp) -> None:
    app_logic.trajectoryZeroReferenceTime = 0.5
    time = np.array([0.0, 1.0, 2.0])
    trajectory = np.array([0.0, 2.0, 4.0])

    zeroed = app_logic.zero_trajectory_to_reference(time, trajectory)

    np.testing.assert_allclose(zeroed, np.array([-1.0, 1.0, 3.0]))


def test_build_nco_power_derived_channels_merges_event_times(app_logic: GUIapp) -> None:
    app_logic.channels = [
        [
            {
                "chanLabel": "NCO_1_am",
                "label": "NCO_1_am",
                "type": "NCO",
                "ind": "1",
                "key": "am",
                "plotType": "mag",
                "units": "%",
                "t": np.array([0.0, 1.0, 3.0]),
                "data": np.array([0.0, 50.0, 100.0]),
                "annotations": [],
            },
        ],
        [
            {
                "chanLabel": "NCO_1_pw",
                "label": "NCO_1_pw",
                "type": "NCO",
                "ind": "1",
                "key": "pw",
                "plotType": "power",
                "units": "W",
                "t": np.array([0.0, 1.0, 2.0, 3.0]),
                "data": np.array([10.0, 20.0, 20.0, 40.0]),
                "annotations": [],
            },
        ],
    ]

    derived_channels = app_logic.build_nco_power_derived_channels()

    assert len(derived_channels) == 3
    output_power = derived_channels[0][0]
    energy = derived_channels[1][0]
    average_power = derived_channels[2][0]

    np.testing.assert_allclose(output_power["t"], np.array([0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(output_power["data"], np.array([0.0, 5.0, 5.0, 40.0]))
    np.testing.assert_allclose(energy["data"], np.array([0.0, 0.0, 5.0, 10.0]))
    np.testing.assert_allclose(average_power["data"], np.array([0.0, 0.0, 1.66666667, 2.5]), rtol=1e-7, atol=1e-7)


def test_gradient_scaling_uses_configured_gamma_for_mt_per_m(app_logic: GUIapp) -> None:
    app_logic.gradientCalibrationHzPerMm = 42.0
    app_logic.nucleusGammaMHzPerT = 21.0
    app_logic.displayGradientsInMtPerM = True

    scaled = app_logic.scale_gradient_data(np.array([0.0, 50.0, 100.0]), "%")

    np.testing.assert_allclose(scaled, np.array([0.0, 1.0, 2.0]))
    assert app_logic.get_gradient_display_units("%") == "mT/m"


def test_set_interaction_mode_clears_shared_measurement_state(app_logic: GUIapp) -> None:
    plot = DummyPlot()
    app_logic.plots = [plot]
    app_logic.measurement_start_x = 0.123
    app_logic.measurement_source_plot = object()
    app_logic.measureSnapToEvents = False
    app_logic.currentCursorTime = None
    app_logic.currentMeasurement = None
    app_logic.windowWidth = 1e-2
    app_logic.update_status = lambda *args, **kwargs: None

    app_logic.setInteractionMode("inspect")

    assert app_logic.measurement_start_x is None
    assert app_logic.measurement_source_plot is None
    assert plot.mode == "inspect"


def test_get_rf_events_parses_line_mapping_and_timeline() -> None:
    pulse_program = {
        "pulseprogram": {
            "@path": "/tmp/test.ppg",
            "@timeunit": "1e-6",
            "linemapping": {
                "map": [
                    {"@ln": "10", "@file": "test.ppg", "@line": "20"},
                    {"@ln": "8000000", "@intern": "DWELL-PROGRAM", "@line": "0"},
                ],
            },
            "ev": [
                {"@t": "0", "@nco": "1", "@am": "0"},
                {"@t": "100", "@ln": "10", "@am": "10"},
                {"@t": "200", "@ln": "8000000", "@am": "5"},
            ],
        },
    }

    _ncos, info = getRFEvents(pulse_program)

    assert info["pulProg"] == "test.ppg"
    assert info["lineMapping"][10]["source"] == "test.ppg"
    assert info["lineMapping"][10]["line"] == 20
    assert info["lineMapping"][8000000]["source"] == "DWELL-PROGRAM"
    np.testing.assert_allclose(info["lineEventTimes"], np.array([100e-6, 200e-6]))
    np.testing.assert_array_equal(info["lineEventNumbers"], np.array([10, 8000000]))


def test_get_pulse_program_location_uses_nearest_previous_event(app_logic: GUIapp) -> None:
    app_logic.pulseProgramTimeline = (
        np.array([0.001, 0.002, 0.003]),
        np.array([185, 193, 8000000]),
    )
    app_logic.pulseProgramLineMapping = {
        185: {"source": "gradMap.ppg", "line": 48},
        193: {"source": "SliceSelection.mod", "line": 40},
        8000000: {"source": "DWELL-PROGRAM", "line": 0},
    }

    assert app_logic.get_pulse_program_location(0.0005) == "-"
    assert app_logic.get_pulse_program_location(0.0024) == "SliceSelection.mod:40 (ln 193)"
    assert app_logic.get_pulse_program_location(0.0031) == "DWELL-PROGRAM:0 (ln 8000000)"
