from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from simView import PROTON_GAMMA_MHZ_PER_T, GUIapp
from utils import multiplot
from utils.simUtilsBrkr import (
    annotate_fcube_event_jobs,
    apply_acquisition_windows_to_rgp,
    build_pulse_program_event_annotations,
    bruker_pw_attenuation_db_to_watts,
    extract_acquisition_windows,
    extract_sample_acquisition_window_details,
    extract_sample_acquisition_windows,
    getRFEvents,
    readGrads,
    readBrkrChannels,
    read_all_fcube_event_infos,
    read_pulse_program_structure,
)
from utils.simUtilsNMRScopeB import readNMRScopeBChannels


def make_app() -> GUIapp:
    app = GUIapp.__new__(GUIapp)
    app.gradientCalibrationHzPerMm = 0.0
    app.nucleusGammaMHzPerT = PROTON_GAMMA_MHZ_PER_T
    app.gradientDisplayUnits = "hz_per_mm"
    app.splitGradientChannels = False
    app.trajectoryZeroReferenceTime = None
    app.trajectoryRefocusTimes = []
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


class DummyCursorPlot:
    def __init__(self) -> None:
        self.markers: list[tuple[float, str, str]] = []
        self.marker_groups: list[str | None] = []
        self.regions: list[tuple[float, float, str | None]] = []

    def add_annotation_marker(
        self,
        time_value: float,
        text_value: str,
        *,
        color: str = "r",
        group: str | None = None,
    ) -> None:
        self.markers.append((time_value, text_value, color))
        self.marker_groups.append(group)

    def clear_annotation_markers(self, *, group: str | None = None) -> None:
        retained = [
            (marker, marker_group)
            for marker, marker_group in zip(self.markers, self.marker_groups, strict=True)
            if group is not None and marker_group != group
        ]
        self.markers = [marker for marker, _group in retained]
        self.marker_groups = [marker_group for _marker, marker_group in retained]

    def add_overlay_region(self, start_time: float, end_time: float, *, color=(50, 140, 255, 45), group=None) -> None:
        self.regions.append((start_time, end_time, group))

    def clear_overlay_regions(self, *, group: str | None = None) -> None:
        self.regions = [region for region in self.regions if group is not None and region[2] != group]


@pytest.fixture
def app_logic() -> GUIapp:
    return make_app()


@pytest.fixture
def bruker_fixture_path() -> Path:
    return Path(__file__).resolve().parents[1] / "testData" / "mrScanSim"


@pytest.fixture
def nmrscopeb_fixture_path() -> Path:
    return Path(__file__).resolve().parents[1] / "testData" / "NMRScopeBData"


def test_read_bruker_channels_from_fixture_loads_gradients_and_ncos(bruker_fixture_path: Path) -> None:
    progress = DummyProgress()
    window = DummyMainWindow()

    channels = readBrkrChannels(str(bruker_fixture_path), progress, window)

    assert channels is not None
    assert len(channels) > 1
    assert "originPPG:" in window.window_title

    gradient_channel = channels[-1]
    assert gradient_channel[0]["chanLabel"] == "Gradients"
    assert [line["key"] for line in gradient_channel[:3]] == ["Gx", "Gy", "Gz"]
    extra_gradient_labels = {line.get("source_attr"): line["key"] for line in gradient_channel[3:]}
    if "g4" in extra_gradient_labels:
        assert extra_gradient_labels["g4"] == "B0"
    assert all(
        key == "B0" if source_attr == "g4" else key == source_attr
        for source_attr, key in extra_gradient_labels.items()
    )
    assert all(line["units"] == "%" for line in gradient_channel)
    assert all(line["raw_units"] == "%" for line in gradient_channel)
    assert all(line["t"].size == gradient_channel[0]["t"].size for line in gradient_channel)

    nco_keys = {channel[0]["key"] for channel in channels[:-1]}
    assert "am" in nco_keys
    assert "pw" in nco_keys
    pw_line = next(channel[0] for channel in channels[:-1] if channel[0]["key"] == "pw")
    assert pw_line["units"] == "W"
    assert pw_line["raw_units"] == "dB"
    assert "raw_data" in pw_line
    np.testing.assert_allclose(
        pw_line["data"],
        bruker_pw_attenuation_db_to_watts(np.asarray(pw_line["raw_data"], dtype=float), reference_watts=1.0),
        rtol=1e-12,
        atol=1e-12,
    )


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


def test_read_grads_supports_extra_gradient_channels(tmp_path: Path) -> None:
    (tmp_path / "_GCube.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<pulseprogram timeunit="0.001">
  <ev t="1" g1="10" g2="20" g3="30" g4="40" g5="50" />
  <ev t="2" g1="11" g2="21" g3="31" g4="41" g5="51" />
</pulseprogram>
""",
        encoding="utf-8",
    )

    grad_time, gradient_attr_names, grads = readGrads(str(tmp_path))

    assert gradient_attr_names == ["g1", "g2", "g3", "g4", "g5"]
    np.testing.assert_allclose(grad_time, np.array([0.001, 0.002]))
    np.testing.assert_allclose(
        grads,
        np.array(
            [
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
                [40.0, 41.0],
                [50.0, 51.0],
            ]
        ),
    )


def test_gradient_duty_cycle_uses_startup_padding(app_logic: GUIapp) -> None:
    time = np.array([0.0, 1.0, 2.0])
    data = np.array([1.0, 1.0, 0.0])

    duty_cycle = app_logic.compute_gradient_duty_cycle(time, data)

    np.testing.assert_allclose(duty_cycle, np.array([0.0, 50.0, 66.66666667]), rtol=1e-7, atol=1e-7)


def test_gradient_trajectory_integrates_sparse_nonequidistant_trapezoids(app_logic: GUIapp) -> None:
    time = np.array([0.0, 0.3, 1.0, 2.0, 2.2, 2.5, 3.0, 3.4, 4.0, 4.8, 5.3, 5.5, 6.0])
    data = np.array([0.0, 6.0, 10.0, 10.0, 6.0, 0.0, 0.0, -4.0, -8.0, -8.0, -3.0, 0.0, 0.0])

    trajectory = app_logic.compute_gradient_trajectory(time, data)

    expected = np.array([0.0, 0.9, 6.5, 16.5, 18.1, 19.0, 19.0, 18.2, 14.6, 8.2, 5.45, 5.15, 5.15])
    np.testing.assert_allclose(trajectory, expected, rtol=1e-12, atol=1e-12)


def test_effective_gradient_changes_sign_at_each_refocus(app_logic: GUIapp) -> None:
    app_logic.trajectoryRefocusTimes = [1.0, 2.0]

    time, effective = app_logic.compute_effective_gradient_profile(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0, 4.0]),
    )

    np.testing.assert_allclose(time, np.array([0.0, 1.0, 1.0, 2.0, 2.0, 3.0]))
    np.testing.assert_allclose(effective, np.array([1.0, 2.0, -2.0, -3.0, 3.0, 4.0]))


def test_b_matrix_at_cursor_integrates_effective_trajectory(app_logic: GUIapp) -> None:
    app_logic.channels = [[{
        "chanLabel": "Effective Trajectory",
        "units": "cycles/mm",
        "t": np.array([0.0, 1.0]),
        "data": np.array([0.0, 1.0]),
    }]]

    matrix = app_logic.compute_b_matrix_at_time(1.0)

    assert matrix is not None
    assert matrix[0, 0] == pytest.approx((2.0 * np.pi) ** 2 / 3.0)
    np.testing.assert_allclose(matrix[1:, :], 0.0)
    np.testing.assert_allclose(matrix[:, 1:], 0.0)


def test_b_matrix_is_zero_before_first_excitation(app_logic: GUIapp) -> None:
    app_logic.channels = [[{
        "chanLabel": "Effective Trajectory",
        "units": "cycles/mm",
        "t": np.array([0.0, 1.0, 2.0]),
        "data": np.array([0.0, 1.0, 0.0]),
    }]]
    app_logic.get_trajectory_excitation_times = lambda _time: np.array([1.5])

    matrix = app_logic.compute_b_matrix_at_time(1.49)

    assert matrix is not None
    np.testing.assert_allclose(matrix, np.zeros((3, 3)))


def test_manual_zero_after_excitation_does_not_hide_b_matrix_origin(app_logic: GUIapp) -> None:
    app_logic.trajectoryZeroReferenceTime = 1.1
    app_logic.channels = [[{
        "chanLabel": "Effective Trajectory",
        "units": "cycles/mm",
        "t": np.array([0.0, 1.0, 1.1, 2.0]),
        "data": np.array([0.0, 0.0, 0.1, 1.0]),
    }]]
    app_logic.get_trajectory_excitation_times = lambda _time: np.array([1.0, 3.0])

    matrix = app_logic.compute_b_matrix_at_time(2.0)

    assert matrix is not None
    assert matrix[0, 0] > 0.0


def test_zero_trajectory_to_reference_interpolates_at_cursor_time(app_logic: GUIapp) -> None:
    app_logic.trajectoryZeroReferenceTime = 0.5
    time = np.array([0.0, 1.0, 2.0])
    trajectory = np.array([0.0, 2.0, 4.0])

    zeroed = app_logic.zero_trajectory_to_reference(time, trajectory)

    np.testing.assert_allclose(zeroed, np.array([-1.0, 1.0, 3.0]))


def test_refocus_trajectory_flips_subsequent_gradient_contributions(app_logic: GUIapp) -> None:
    app_logic.trajectoryRefocusTimes = [1.0]
    time = np.array([0.0, 1.0, 2.0, 3.0])
    trajectory = np.array([0.0, 2.0, 4.0, 6.0])

    refocused = app_logic.apply_trajectory_refocuses(time, trajectory)

    np.testing.assert_allclose(refocused, np.array([0.0, 2.0, 0.0, -2.0]))


def test_trajectory_display_transforms_apply_zero_before_refocus(app_logic: GUIapp) -> None:
    app_logic.trajectoryZeroReferenceTime = 0.5
    app_logic.trajectoryRefocusTimes = [1.0]
    time = np.array([0.0, 1.0, 2.0])
    trajectory = np.array([0.0, 2.0, 4.0])

    transformed = app_logic.apply_trajectory_display_transforms(time, trajectory)

    np.testing.assert_allclose(transformed, np.array([-1.0, 1.0, -1.0]))


def test_refocus_trajectory_uses_selected_zero_as_coherence_path_origin(app_logic: GUIapp) -> None:
    app_logic.trajectoryZeroReferenceTime = 1.0
    app_logic.trajectoryRefocusTimes = [0.5, 2.0]
    time = np.array([0.0, 1.0, 2.0, 3.0])
    trajectory = np.array([0.0, 2.0, 4.0, 6.0])

    transformed = app_logic.apply_trajectory_display_transforms(time, trajectory)

    np.testing.assert_allclose(transformed, np.array([0.0, 0.0, 2.0, 0.0]))


def test_refocus_trajectory_handles_multiple_off_sample_flips(app_logic: GUIapp) -> None:
    app_logic.trajectoryZeroReferenceTime = 0.0
    app_logic.trajectoryRefocusTimes = [0.5, 1.5, 2.5]
    time = np.array([0.0, 1.0, 2.0, 3.0])
    trajectory = np.array([0.0, 1.0, 2.0, 3.0])

    transformed = app_logic.apply_trajectory_display_transforms(time, trajectory)

    np.testing.assert_allclose(transformed, np.array([0.0, 0.0, 0.0, 0.0]))


def test_trajectory_resets_at_each_calibrated_excitation_block(app_logic: GUIapp) -> None:
    app_logic.rfPulseCalibrations = [{"name": "ExcPulse1", "duration": 0.5, "flip_angle": 90.0}]
    app_logic.detect_rf_pulse_descriptors = lambda: [
        {"focus": 2.0, "duration": 0.5}, {"focus": 4.0, "duration": 0.5},
    ]
    time = np.arange(6.0)
    trajectory = np.arange(6.0)

    transformed = app_logic.apply_trajectory_refocuses(time, trajectory)

    np.testing.assert_allclose(transformed, np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))


@pytest.mark.parametrize(
    "name",
    ["PVM_FovSatExcPulse", "PVM_FatSatExcPulse", "PVM_FatSupPulse", "PVM_MTPulse", "PrepPulse1"],
)
def test_preparation_pulse_names_are_not_trajectory_excitations(app_logic: GUIapp, name: str) -> None:
    calibration = {"name": name, "duration": 0.5, "flip_angle": 90.0}

    assert not app_logic.is_trajectory_excitation_calibration(calibration)


def test_only_named_sequence_excitation_resets_trajectory(app_logic: GUIapp) -> None:
    app_logic.rfPulseCalibrations = [
        {"name": "PVM_FovSatExcPulse", "duration": 0.2, "flip_angle": 90.0},
        {"name": "PVM_FatSatExcPulse", "duration": 0.3, "flip_angle": 90.0},
        {"name": "ExcPulse1", "duration": 0.5, "flip_angle": 90.0},
    ]
    app_logic.detect_rf_pulse_descriptors = lambda: [
        {"focus": 1.0, "duration": 0.2},
        {"focus": 2.0, "duration": 0.3},
        {"focus": 3.0, "duration": 0.5},
    ]

    reset_times = app_logic.get_trajectory_excitation_times(np.array([0.0, 4.0]))

    np.testing.assert_allclose(reset_times, np.array([3.0]))


def test_multi_lobe_excitation_is_matched_as_one_calibrated_pulse(app_logic: GUIapp) -> None:
    app_logic.rfPulseCalibrations = [
        {"name": "ExcPul", "duration": 2.7, "flip_angle": 90.0},
    ]
    app_logic.detect_rf_pulse_descriptors = lambda: [
        {"start": 1.0, "end": 1.9, "focus": 1.45, "duration": 0.9, "nco": "1"},
        {"start": 1.91, "end": 2.79, "focus": 2.35, "duration": 0.88, "nco": "1"},
        {"start": 2.80, "end": 3.70, "focus": 3.25, "duration": 0.9, "nco": "1"},
    ]

    reset_times = app_logic.get_trajectory_excitation_times(np.array([0.0, 5.0]))

    np.testing.assert_allclose(reset_times, np.array([2.35]))


def test_coherence_order_toggles_at_flips_relative_to_selected_zero(app_logic: GUIapp) -> None:
    app_logic.trajectoryZeroReferenceTime = 1.0
    app_logic.trajectoryRefocusTimes = [0.5, 2.0]

    profile_time, coherence_order = app_logic.compute_coherence_order_profile(np.array([0.0, 3.0]))

    np.testing.assert_allclose(profile_time, np.array([0.0, 0.5, 2.0, 3.0]))
    np.testing.assert_allclose(coherence_order, np.array([1.0, -1.0, 1.0, 1.0]))


def test_imperfect_refocus_pathway_weights_branch_transverse_and_longitudinal_states(app_logic: GUIapp) -> None:
    app_logic.trajectoryRefocusTimes = [1.0]
    app_logic.trajectoryRefocusFlipAngleDegrees = 180.0

    time, weights = app_logic.compute_imperfect_refocus_pathway_weights(np.array([0.0, 2.0]))

    np.testing.assert_allclose(time, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(weights[:, 0], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(weights[:, 1], np.array([0.0, 1.0, 0.0]), atol=1e-12)

    app_logic.trajectoryRefocusFlipAngleDegrees = 170.0
    _time, imperfect_weights = app_logic.compute_imperfect_refocus_pathway_weights(np.array([0.0, 2.0]))

    assert imperfect_weights[0, 1] > 0.0
    assert imperfect_weights[1, 1] > 0.0
    assert imperfect_weights[2, 1] > 0.0


def test_imperfect_coherence_branches_keep_distinct_rf_histories(app_logic: GUIapp) -> None:
    app_logic.trajectoryRefocusTimes = [1.0, 2.0]
    app_logic.trajectoryRefocusFlipAngleDegrees = 170.0

    time, branches = app_logic.compute_imperfect_coherence_branches(np.array([0.0, 3.0]))

    assert np.array_equal(time, np.array([0.0, 1.0, 2.0, 3.0]))
    assert len(branches) > 1
    assert all(len(branch["data"]) == len(time) for branch in branches)
    assert len({branch["label"] for branch in branches}) == len(branches)


def test_coherence_channel_keeps_only_gradient_time_extent(app_logic: GUIapp) -> None:
    gradient_time = np.linspace(0.0, 10.0, 100_000)
    app_logic.channels = [
        [
            {
                "chanLabel": "Gradients",
                "label": "Gx",
                "type": "grads",
                "ind": "0",
                "key": "Gx",
                "plotType": "mag",
                "units": "%",
                "t": gradient_time,
                "data": np.zeros_like(gradient_time),
                "annotations": [],
            },
        ],
    ]

    derived_channels = app_logic.build_gradient_derived_channels()
    coherence_line = next(channel[0] for channel in derived_channels if channel[0]["chanLabel"] == "Coherence Order")

    np.testing.assert_allclose(coherence_line["source_time"], np.array([0.0, 10.0]))
    assert coherence_line["pen"] in multiplot.makePens()[1]


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


def test_bruker_pw_attenuation_db_to_watts_converts_from_db() -> None:
    attenuation_db = np.array([0.0, 3.0, 10.0, 20.0])
    watts = bruker_pw_attenuation_db_to_watts(attenuation_db, reference_watts=1.0)
    expected = np.array([1.0, 10 ** (-0.3), 0.1, 0.01])
    np.testing.assert_allclose(watts, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        bruker_pw_attenuation_db_to_watts(np.array([10.0]), reference_watts=100.0),
        np.array([10.0]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_get_rf_events_treats_zero_rgp_as_inactive() -> None:
    pulse_program = {
        "pulseprogram": {
            "@path": "/tmp/test.ppg",
            "@timeunit": "1.0",
            "ev": [
                {"@t": "0.0", "@nco": "3", "@rgp": "0--0x1"},
                {"@t": "1.0", "@nco": "3", "@rgp": "0--0"},
            ],
        },
    }

    ncos, _info = getRFEvents(pulse_program)

    np.testing.assert_allclose(ncos["3"]["rgp"], np.array([1.0, 0.0]))


def test_extract_acquisition_windows_uses_fcube_write_event_as_end() -> None:
    event_infos = [
        {
            "fcube": "_FCube1",
            "events": [
                {"t": 10.0, "nco": "3", "rgp": "0--0x1"},
                {"t": 10.0001, "nco": "3", "rgp": "0--0"},
            ],
        },
        {
            "fcube": "_FCube4",
            "events": [
                {"t": 10.0, "nco": "3", "rgp": "0--0x1"},
                {"t": 10.0001, "nco": "3", "rgp": "0--0"},
                {"t": 11.5, "nco": "0", "wr": "0", "if": "0"},
                {"t": 20.0, "nco": "3", "rgp": "0--0x1"},
                {"t": 20.0001, "nco": "3", "rgp": "0--0"},
            ],
        },
    ]

    windows = extract_acquisition_windows(event_infos)

    assert windows == {"3": [(10.0, 11.5), (20.0, 21.5)]}


def test_extract_sample_acquisition_windows_uses_internal_dwell_end() -> None:
    event_infos = [
        {
            "fcube": "_FCube4",
            "events": [
                {"t": 10.0, "nco": "3", "rgp": "0--0x1", "job": "0"},
                {"t": 10.0001, "nco": "3", "rgp": "0--0", "job": "0"},
                {"t": 11.0, "nco": "3", "ln": "10000001", "job": "0"},
                {"t": 11.5, "nco": "0", "wr": "0", "if": "0", "job": "0"},
            ],
        },
    ]

    windows = extract_sample_acquisition_windows(event_infos)

    assert windows == {"3": [(10.0, 11.0)]}


def test_extract_sample_acquisition_windows_accepts_bruker_11000001_end_line() -> None:
    event_infos = [{"events": [
        {"t": 10.0, "nco": "3", "rgp": "0--0x1"},
        {"t": 11.0, "ln": "11000001"},
    ]}]

    assert extract_sample_acquisition_windows(event_infos) == {"3": [(10.0, 11.0)]}


def test_sample_acquisition_window_details_retains_job_type() -> None:
    event_infos = [{"events": [
        {"t": 10.0, "nco": "3", "rgp": "0--0x1", "ln": "13000000"},
        {"t": 11.0, "nco": "3", "ln": "13000001"},
    ]}]

    assert extract_sample_acquisition_window_details(event_infos) == [
        {"nco": "3", "start": 10.0, "end": 11.0, "job_type": "job 13"},
    ]


def test_annotate_fcube_event_jobs_carries_job_to_newer_sample_end_line() -> None:
    info = {"events": [
        {"t": 10.0, "ln": "13000000", "rgp": "0--0x1"},
        {"t": 11.0, "ln": "13000001"},
    ]}

    annotate_fcube_event_jobs(info, {"13000000": "7"})

    assert info["events"][1]["job"] == "7"


def test_acquisition_windows_match_starts_and_ends_by_job_number() -> None:
    event_infos = [
        {
            "fcube": "_FCube4",
            "events": [
                {"t": 10.0, "nco": "3", "rgp": "0--0x1", "job": "0"},
                {"t": 10.1, "nco": "3", "rgp": "0--0x1", "job": "1"},
                {"t": 10.2, "nco": "0", "wr": "0", "if": "0", "job": "1"},
                {"t": 10.3, "nco": "0", "wr": "0", "if": "0", "job": "0"},
            ],
        },
    ]

    windows = extract_acquisition_windows(event_infos)

    assert windows == {"3": [(10.0, 10.3), (10.1, 10.2)]}


def test_annotate_fcube_event_jobs_carries_job_to_internal_rgp_events() -> None:
    info = {
        "events": [
            {"t": 1.0, "ln": "1323", "nco": "3"},
            {"t": 1.1, "ln": "10000000", "nco": "3", "rgp": "0--0x1"},
            {"t": 1.2, "ln": "1342", "nco": "0", "wr": "0", "if": "0"},
        ],
    }

    annotate_fcube_event_jobs(info, {"1323": "0", "1342": "0"})

    assert [event.get("job") for event in info["events"]] == ["0", "0", "0"]


def test_build_pulse_program_event_annotations_displays_job() -> None:
    annotations = build_pulse_program_event_annotations(
        [
            {
                "fcube": "_FCube4",
                "events": [
                    {"t": 1.0, "wr": "0", "if": "0", "job": "0"},
                ],
            },
        ],
    )

    assert annotations[0]["texts"] == ["wr=0 | if=0 | job=0"]


def test_build_pulse_program_event_annotations_omits_standalone_job_labels() -> None:
    annotations = build_pulse_program_event_annotations(
        [
            {
                "fcube": "_FCube4",
                "events": [
                    {"t": 1.0, "job": "0"},
                    {"t": 2.0, "ln": "10000001", "job": "0"},
                    {"t": 3.0, "rgp": "0--0x1", "job": "0"},
                ],
            },
        ],
    )

    assert annotations[0]["texts"] == ["acq=end | job=0", "rgp=on | job=0"]


def test_apply_acquisition_windows_to_rgp_keeps_nco_timebase() -> None:
    ncos = {
        "3": {
            "t": np.array([0.0, 1.0, 2.0]),
            "am": np.array([0.0, 1.0, 0.0]),
            "rgp": np.array([0.0, 0.0, 0.0]),
        },
    }

    apply_acquisition_windows_to_rgp(ncos, {"3": [(10.0, 11.5), (20.0, 21.5)]})

    np.testing.assert_allclose(ncos["3"]["t"], np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(ncos["3"]["rgp_t"], np.array([10.0, 11.5, 20.0, 21.5]))
    np.testing.assert_allclose(ncos["3"]["rgp"], np.array([1.0, 0.0, 1.0, 0.0]))


def test_gradient_scaling_uses_configured_gamma_for_mt_per_m(app_logic: GUIapp) -> None:
    app_logic.gradientCalibrationHzPerMm = 42.0
    app_logic.nucleusGammaMHzPerT = 21.0
    app_logic.gradientDisplayUnits = "mt_per_m"

    scaled = app_logic.scale_gradient_data(np.array([0.0, 50.0, 100.0]), "%")

    np.testing.assert_allclose(scaled, np.array([0.0, 1.0, 2.0]))
    assert app_logic.get_gradient_display_units("%") == "mT/m"


def test_gradient_update_does_not_scale_non_axis_gradient_channels(app_logic: GUIapp) -> None:
    app_logic.gradientCalibrationHzPerMm = 100.0
    app_logic.nucleusGammaMHzPerT = 50.0
    app_logic.gradientDisplayUnits = "mt_per_m"
    app_logic.channels = [
        [
            {
                "type": "grads",
                "key": "Gx",
                "units": "%",
                "raw_units": "%",
                "data": np.array([0.0, 50.0, 100.0]),
            },
            {
                "type": "grads",
                "key": "B0",
                "source_attr": "g4",
                "units": "%",
                "raw_units": "%",
                "data": np.array([10.0, 20.0, 30.0]),
            },
        ],
    ]

    app_logic.update_gradient_channels()

    gx_line, b0_line = app_logic.channels[0]
    np.testing.assert_allclose(gx_line["data"], np.array([0.0, 1.0, 2.0]))
    assert gx_line["units"] == "mT/m"
    np.testing.assert_allclose(b0_line["data"], np.array([10.0, 20.0, 30.0]))
    assert b0_line["units"] == "%"
    assert b0_line["physical_hz_per_mm"] is None


def test_remove_zero_gradient_lines_filters_only_zero_gradient_traces(app_logic: GUIapp) -> None:
    app_logic.channels = [
        [
            {
                "type": "grads",
                "key": "Gx",
                "label": "Gx",
                "raw_data": np.array([0.0, 0.0, 0.0]),
                "data": np.array([0.0, 0.0, 0.0]),
            },
            {
                "type": "grads",
                "key": "Gy",
                "label": "Gy",
                "raw_data": np.array([0.0, 1.0, 0.0]),
                "data": np.array([0.0, 1.0, 0.0]),
            },
        ],
        [
            {
                "type": "NCO",
                "key": "am",
                "data": np.array([0.0, 0.0, 0.0]),
            },
        ],
    ]

    app_logic.remove_zero_gradient_lines()

    assert [line["key"] for line in app_logic.channels[0]] == ["Gy"]
    assert app_logic.channels[1][0]["key"] == "am"


def test_split_recorded_gradient_channels_creates_one_channel_per_gradient(app_logic: GUIapp) -> None:
    app_logic.splitGradientChannels = True
    app_logic.channels = [
        [
            {"type": "NCO", "key": "am", "chanLabel": "NCO_0_am"},
        ],
        [
            {"type": "grads", "key": "Gx", "label": "Gx", "chanLabel": "Gradients"},
            {"type": "grads", "key": "Gy", "label": "Gy"},
        ],
    ]

    app_logic.split_recorded_gradient_channels()

    assert [channel[0]["chanLabel"] for channel in app_logic.channels] == ["NCO_0_am", "Gx", "Gy"]
    assert all(len(channel) == 1 for channel in app_logic.channels)


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
    assert len(info["events"]) == 3
    assert info["events"][1]["ln"] == "10"
    assert np.isclose(float(info["events"][2]["t"]), 200e-6)


def test_output_structure_keeps_unexecuted_ppg_loops(tmp_path: Path) -> None:
    (tmp_path / "_FCube1.output").write_text(
        "loopcounters: 256 items\n=======================\n 0 3 0\n\n\n"
        "delay increments [sec]: 64 items\n"
        " l1:  14   1  0  1 in lines: 12\n"
        " 10 \"'demo.ppg': line 10---> echo, d3 grad_ramp{1, 0, 0}\"\n"
        " 11 \"'demo.ppg': line 11---> (p1:sp1 ph1):f1\"\n"
        " 12 \"'demo.ppg': line 12---> lo to echo times l1\"\n",
    )

    structure = read_pulse_program_structure(str(tmp_path), "demo.ppg")

    assert structure["loop_values"]["l1"] == 3
    assert [record["line"] for record in structure["records"]] == [10, 11, 12]


def test_output_structure_prefers_exact_precompiled_ppg_with_modules(tmp_path: Path) -> None:
    module_path = tmp_path / "pulseprogram.precomp"
    module_path.write_text(
        '# 10 "/pp/demo.ppg"\n'
        'start, 10u\n'
        ';EXEC_begin Prep Prep()\n'
        '# 4 "/pp/Prep.mod" 1\n'
        '20u grad_ramp{g1, 0, 0}\n'
        ';EXEC_end Prep\n'
        '# 11 "/pp/demo.ppg" 2\n'
        '(p0:sp0 ph0):f1\n'
    )
    (tmp_path / "_FCube1.output").write_text(f"pulse program: {module_path}\n")

    structure = read_pulse_program_structure(str(tmp_path), "demo.ppg")

    assert structure["precomp"] == str(module_path)
    assert [(record["source"], record["line"]) for record in structure["records"]] == [
        ("demo.ppg", 10), ("demo.ppg", 11), ("Prep.mod", 4),
        ("Prep.mod", 5), ("demo.ppg", 11),
    ]
    assert structure["records"][2]["text"] == "20u grad_ramp{g1, 0, 0}"


def test_output_structure_reads_precalculated_gradient_ramps(tmp_path: Path) -> None:
    (tmp_path / "_FCube1.output").write_text("")
    (tmp_path / "_GCube.output").write_text(
        "Gcon: Ramp Shape Values\n"
        "-----------------------\n\n"
        "ramp    0: 3 items\n"
        "------------------\n"
        " 0.25 0.50 0.75\n\n"
        "ramp    1: 2 items\n"
        "------------------\n"
        " 0.33 0.66\n\n"
        "pulse increments [usec]: 64 items\n"
    )

    structure = read_pulse_program_structure(str(tmp_path), "demo.ppg")

    assert structure["gradient_ramps"] == [
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [0.0, 0.33, 0.66, 1.0],
    ]
    assert structure["gradient_ramp_time_source"] == "Gcon raster fallback"


def test_full_ppg_uses_method_loop_counts_and_omits_subroutine_calls(app_logic: GUIapp) -> None:
    app_logic.pulseProgramStructure = {
        "records": [
            {"source": "demo.ppg", "line": 1, "text": "define loopcounter Echoes = {$PVM_RareFactor}"},
            {"source": "demo.ppg", "line": 2, "text": "echo, 10u"},
            {"source": "demo.ppg", "line": 3, "text": "subr DisabledModule"},
            {"source": "demo.ppg", "line": 4, "text": "lo to echo times Echoes"},
        ],
        "parameters": {"PVM_RareFactor": 8.0},
        "loop_values": {},
    }

    events = app_logic.build_full_ppg_schematic_events()

    assert not any(event["kind"] == "subroutine" for event in events)
    assert next(event for event in events if event["kind"] == "loop")["count"] == 8


def test_ppg_duration_resolves_d_array_and_defined_delay_math(app_logic: GUIapp) -> None:
    app_logic.pulseProgramStructure = {"parameters": {"d3": 144e-6, "DE": 10e-6}}

    assert np.isclose(app_logic.resolve_ppg_duration("d3"), 144e-6)
    assert np.isclose(
        app_logic.resolve_ppg_duration("denab", {"denab": "d3 - de"}),
        134e-6,
    )
    assert np.isclose(
        app_logic.resolve_ppg_duration("d5m40u", {"d5m40u": "d3 - 40u"}),
        104e-6,
    )


def test_ppg_condition_resolves_paravision_diffusion_mode_enum(app_logic: GUIapp) -> None:
    parameters = {"PVM_DiffPrepMode": "SpinEcho"}

    assert app_logic.evaluate_ppg_condition("PVM_DiffPrepMode == 0", parameters) is True
    assert app_logic.evaluate_ppg_condition("PVM_DiffPrepMode == 1", parameters) is False


def test_get_rf_events_captures_all_given_event_attributes() -> None:
    pulse_program = {
        "pulseprogram": {
            "@path": "/tmp/test.ppg",
            "@timeunit": "1e-6",
            "ev": [
                {
                    "@t": "12",
                    "@ln": "42",
                    "@seq": "cwait",
                    "@tr": "10",
                    "@wl": "3",
                    "@g": "6",
                    "@nco": "2",
                    "@pw": "27.9",
                    "@ncoc": "9",
                    "@sf": "400.322",
                    "@scanidx": "7",
                    "@scanph": "90",
                    "@scancmd": "st",
                    "@wr": "1",
                    "@if": "2",
                    "@df": "3",
                    "@rf": "1",
                    "@rgp": "0--0x1",
                    "@p0": "0",
                    "@p1": "90",
                    "@p2": "180",
                    "@am": "50",
                },
            ],
        },
    }

    _ncos, info = getRFEvents(pulse_program)

    assert len(info["events"]) == 1
    event = info["events"][0]
    expected_keys = {
        "t",
        "ln",
        "seq",
        "tr",
        "wl",
        "g",
        "nco",
        "pw",
        "ncoc",
        "sf",
        "scanidx",
        "scanph",
        "scancmd",
        "wr",
        "if",
        "df",
        "rf",
        "rgp",
        "p0",
        "p1",
        "p2",
        "am",
    }
    assert expected_keys.issubset(event.keys())
    assert event["seq"] == "cwait"
    assert event["rgp"] == "0--0x1"


def test_build_pulse_program_event_annotations_extracts_control_events() -> None:
    event_infos = [
        {
            "fcube": "FCube1",
            "events": [
                {"t": 0.001, "nco": "1", "am": "5"},
                {"t": 0.002, "seq": "cwait", "tr": "10", "ln": "185"},
                {"t": 0.002, "seq": "cwait", "tr": "10", "ln": "185"},
            ],
        },
        {
            "fcube": "FCube4",
            "events": [
                {"t": 0.002, "seq": "cwait", "tr": "10", "ln": "185"},
                {"t": 0.003, "scancmd": "st", "wr": "1", "ln": "191"},
            ],
        },
    ]

    annotations = build_pulse_program_event_annotations(event_infos)

    assert len(annotations) == 1
    annotation = annotations[0]
    np.testing.assert_allclose(annotation["t"], np.array([0.002, 0.003]))
    assert annotation["texts"][0] == "seq=cwait | tr=10"
    assert annotation["texts"][1] == "scan=st | wr=1"
    assert "scan=st" in annotation["texts"][1]


def test_add_annotations_supports_text_labels() -> None:
    line = {
        "annotations": [
            {
                "t": np.array([0.001, 0.002]),
                "texts": ["seq=cwait | tr=10", "scan=st | wr=1"],
                "color": "y",
            },
        ],
    }
    plot = DummyCursorPlot()

    multiplot.addAnnotations(line, plot)

    assert len(plot.markers) == 2
    assert plot.markers[0] == (0.001, "seq=cwait | tr=10", "y")
    assert plot.markers[1] == (0.002, "scan=st | wr=1", "y")


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


def test_sequence_outline_nests_ppg_modules_and_gradients_in_cycles(app_logic: GUIapp) -> None:
    app_logic.pulseProgramTimeline = (
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        np.array([10, 11, 20, 10, 11]),
    )
    app_logic.pulseProgramLineMapping = {
        10: {"source": "main.ppg", "line": 10},
        11: {"source": "main.ppg", "line": 11},
        20: {"source": "Slice.mod", "line": 5},
    }
    app_logic.channels = [[{
        "type": "grads", "key": "Gx", "t": np.array([0.0, 1.5, 2.5, 4.0]),
        "raw_data": np.array([0.0, 20.0, 0.0, 0.0]),
    }]]
    app_logic.get_trajectory_excitation_times = lambda _time: np.array([1.0, 3.0])

    outline = app_logic.build_sequence_outline()

    assert [cycle["label"] for cycle in outline] == ["Preparation", "Cycle 1", "Cycle 2"]
    assert [module["source"] for module in outline[1]["modules"]] == ["main.ppg", "Slice.mod"]
    gradient_descriptions = [
        gradient["description"]
        for module in outline[1]["modules"]
        for event in module["events"]
        for gradient in event["gradients"]
    ]
    assert gradient_descriptions == ["Gx=20%", "Gx=0%"]
    assert outline[1]["modules"][0]["events"][0]["delay"] == pytest.approx(1.0)


def test_read_nmrscopeb_channels_accepts_list_based_time_payload() -> None:
    data = {
        "time": [0.0, 1.0, 2.0],
        "rf_am": {"val": [10.0, 20.0, 30.0], "units": "%", "show": "yes"},
        "gx": {"val": [1.0, 2.0, 3.0], "units": "%", "show": "no"},
    }

    channels = readNMRScopeBChannels(data, DummyProgress(), DummyMainWindow())

    assert len(channels) == 2
    rf_channel = channels[0][0]
    gx_channel = channels[1][0]
    np.testing.assert_allclose(rf_channel["t"], np.array([0.0, 0.001, 0.002]))
    assert rf_channel["show"] is True
    assert gx_channel["show"] is False


def test_read_nmrscopeb_channels_from_fixture_path(nmrscopeb_fixture_path: Path) -> None:
    channels = readNMRScopeBChannels(str(nmrscopeb_fixture_path), DummyProgress(), DummyMainWindow())

    assert channels is not None
    assert len(channels) > 0
    labels = [channel[0]["label"] for channel in channels]
    assert "gx" in labels
    assert "gy" in labels
    assert "gz" in labels

    gx_channel = next(channel[0] for channel in channels if channel[0]["label"] == "gx")
    assert gx_channel["units"] == ""
    assert gx_channel["chanLabel"].endswith("(-)")
    assert gx_channel["t"].size == gx_channel["data"].size


def test_read_bruker_channels_attaches_pulse_program_event_annotations(bruker_fixture_path: Path) -> None:
    channels = readBrkrChannels(str(bruker_fixture_path), DummyProgress(), DummyMainWindow())

    assert channels is not None
    events_channel = next(channel[0] for channel in channels if channel[0]["chanLabel"] == "NCO_0_events")
    event_annotation = next((ann for ann in events_channel["annotations"] if ann.get("name") == "ppg_events"), None)
    assert event_annotation is not None
    assert len(event_annotation["t"]) > 0
    assert len(event_annotation["texts"]) > 0
    assert any("wr=" in text for text in event_annotation["texts"])

    regular_nco_channel = next(channel[0] for channel in channels if channel[0]["chanLabel"] == "NCO_1_am")
    assert all(ann.get("name") != "ppg_events" for ann in regular_nco_channel["annotations"])

    gradient_channel = channels[-1]
    assert len(gradient_channel) >= 3
    assert [line["key"] for line in gradient_channel[:3]] == ["Gx", "Gy", "Gz"]
    assert not any(ann.get("name") == "ppg_events" for ann in gradient_channel[0]["annotations"])
    assert gradient_channel[1]["annotations"] == []
    assert gradient_channel[2]["annotations"] == []


def test_read_all_fcube_event_infos_parses_all_xml_events(bruker_fixture_path: Path) -> None:
    event_infos = read_all_fcube_event_infos(str(bruker_fixture_path))

    expected_counts: dict[str, int] = {}
    for fcube_path in sorted(bruker_fixture_path.glob("_FCube*.xml")):
        root = ET.parse(fcube_path).getroot()
        expected_counts[fcube_path.stem] = len(root.findall(".//ev"))

    parsed_counts = {
        str(info["fcube"]): len(info.get("events", []))
        for info in event_infos
    }

    assert parsed_counts == expected_counts
    assert sum(parsed_counts.values()) == sum(expected_counts.values())


def test_read_all_fcube_event_infos_preserves_all_event_attributes(bruker_fixture_path: Path) -> None:
    event_infos = read_all_fcube_event_infos(str(bruker_fixture_path))
    parsed_infos = {str(info["fcube"]): info for info in event_infos}

    for fcube_path in sorted(bruker_fixture_path.glob("_FCube*.xml")):
        root = ET.parse(fcube_path).getroot()
        time_unit = float(root.attrib["timeunit"])
        xml_events = root.findall(".//ev")
        parsed_events = parsed_infos[fcube_path.stem]["events"]

        assert len(parsed_events) == len(xml_events)

        for xml_event, parsed_event in zip(xml_events, parsed_events, strict=True):
            normalized_expected_keys = set(xml_event.attrib.keys())
            parsed_keys = set(parsed_event.keys())
            parsed_keys.discard("job")
            assert parsed_keys == normalized_expected_keys

            if "t" in xml_event.attrib:
                assert np.isclose(float(parsed_event["t"]), float(xml_event.attrib["t"]) * time_unit)

            for attr_name, attr_value in xml_event.attrib.items():
                if attr_name == "t":
                    continue
                assert str(parsed_event[attr_name]) == attr_value


def test_detect_rf_pulse_starts_from_nco_channels(app_logic: GUIapp) -> None:
    app_logic.channels = [
        [
            {
                "type": "NCO",
                "key": "am",
                "t": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                "data": np.array([0.0, 10.0, 10.0, 0.0, 5.0]),
            },
        ],
        [
            {
                "type": "NCO",
                "key": "rgp",
                "t": np.array([0.0, 0.5, 1.5]),
                "data": np.array([0.0, 1.0, 0.0]),
            },
        ],
    ]

    starts = app_logic.detect_rf_pulse_starts()
    window_starts, focus_times = app_logic.detect_rf_pulse_windows()

    np.testing.assert_allclose(starts, np.array([1.0, 4.0]))
    np.testing.assert_allclose(window_starts, np.array([1.0, 4.0]))
    np.testing.assert_allclose(focus_times, np.array([2.0, 4.0]))


def test_rf_focus_markers_are_added_only_to_amplitude_channels(app_logic: GUIapp) -> None:
    app_logic.rfPulseFocusTimes = np.array([1.5])
    app_logic.channels = [
        [{"type": "NCO", "key": "am"}],
        [{"type": "NCO", "key": "pw"}],
        [{"type": "grads", "key": "Gx"}],
    ]
    app_logic.plots = [DummyCursorPlot(), DummyCursorPlot(), DummyCursorPlot()]

    app_logic.add_rf_pulse_focus_markers()

    assert app_logic.plots[0].markers == [(1.5, "RF focus", "m")]
    assert app_logic.plots[1].markers == []
    assert app_logic.plots[2].markers == []


def test_detect_rf_pulse_descriptors_records_repeatable_pulse_signatures(app_logic: GUIapp) -> None:
    app_logic.channels = [[{
        "type": "NCO", "key": "am", "ind": "1",
        "t": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "data": np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
    }]]

    pulses = app_logic.detect_rf_pulse_descriptors()

    assert len(pulses) == 2
    assert pulses[0]["duration"] == pulses[1]["duration"] == 1.0
    assert pulses[0]["area"] == pulses[1]["area"] == 0.5


def test_calibrated_refocus_sources_separate_diffusion_and_rare_train(app_logic: GUIapp) -> None:
    app_logic.rfPulseCalibrations = [
        {"name": "PVM_DwRfcPulse1", "duration": 1.7e-3, "flip_angle": 180.0},
        {"name": "ExcPulse1", "duration": 2.1e-3, "flip_angle": 90.0},
        {"name": "RefPulse1", "duration": 1.7e-3, "flip_angle": 180.0},
    ]
    app_logic.detect_rf_pulse_descriptors = lambda: [
        {"focus": 1.0, "duration": 2.1e-3},
        {"focus": 2.0, "duration": 1.7e-3},
        {"focus": 3.0, "duration": 1.7e-3},
        {"focus": 4.0, "duration": 1.7e-3},
    ]

    app_logic.apply_calibrated_refocus_flips()

    assert app_logic.trajectoryFlipSources[2.0] == "Method: PVM_DwRfcPulse1 (180°)"
    assert app_logic.trajectoryFlipSources[3.0] == "Method: RefPulse1 (180°)"
    assert app_logic.trajectoryFlipSources[4.0] == "Method: RefPulse1 (180°)"


def test_trajectory_flip_markers_are_refreshed_only_on_trajectory_plots(app_logic: GUIapp) -> None:
    app_logic.trajectoryRefocusTimes = [2.0, 1.0]
    app_logic.channels = [
        [{"chanLabel": "Gradients"}],
        [{"chanLabel": "Gradient Trajectory"}],
    ]
    app_logic.plots = [DummyCursorPlot(), DummyCursorPlot()]
    app_logic.plots[1].add_annotation_marker(0.5, "old flip", group="trajectory_flip")

    app_logic.refresh_trajectory_flip_markers()

    assert app_logic.plots[0].markers == []
    assert app_logic.plots[1].markers == [
        (1.0, "180° flip", "#006b6b"),
        (2.0, "180° flip", "#006b6b"),
    ]


def test_trajectory_echo_and_acquisition_window_are_marked(app_logic: GUIapp) -> None:
    app_logic.trajectoryZeroReferenceTime = 0.0
    app_logic.trajectoryRefocusTimes = [0.5]
    app_logic.channels = [
        [
            {
                "chanLabel": "Gradient Trajectory",
                "t": np.array([0.0, 1.0, 2.0, 3.0]),
                "data": np.array([0.0, 1.0, 0.0, 1.0]),
            },
        ],
        [
            {
                "chanLabel": "NCO_1_rgp",
                "key": "rgp",
                "t": np.array([1.5, 2.5]),
                "data": np.array([1.0, 0.0]),
            },
        ],
    ]
    app_logic.plots = [DummyCursorPlot(), DummyCursorPlot()]

    app_logic.refresh_trajectory_flip_markers()

    assert app_logic.detect_acquisition_windows() == [(1.5, 2.5)]
    assert app_logic.detect_trajectory_echoes() == [2.0]
    assert app_logic.plots[0].regions == [(1.5, 2.5, "acquisition_window")]
    assert (2.0, "Echo in ADC", "#006b3c") in app_logic.plots[0].markers


def test_spin_echo_is_predicted_from_90_180_pair(app_logic: GUIapp) -> None:
    app_logic.channels = [[{"chanLabel": "Effective Trajectory", "t": np.array([0.0, 10.0])}]]
    app_logic.trajectoryRefocusTimes = [4.0]
    app_logic.get_trajectory_excitation_times = lambda _time: np.array([1.0])

    candidates = app_logic.find_spin_echo_candidates()

    assert candidates == [{"time": 7.0, "excitation_time": 1.0, "refocus_time": 4.0}]


def test_spin_echo_does_not_cross_into_next_excitation_block(app_logic: GUIapp) -> None:
    app_logic.channels = [[{"chanLabel": "Effective Trajectory", "t": np.array([0.0, 10.0])}]]
    app_logic.trajectoryRefocusTimes = [4.0]
    app_logic.get_trajectory_excitation_times = lambda _time: np.array([1.0, 6.0])

    assert app_logic.find_spin_echo_candidates() == []


def test_spin_echo_uses_only_first_180_after_each_excitation(app_logic: GUIapp) -> None:
    app_logic.channels = [[{"chanLabel": "Effective Trajectory", "t": np.array([0.0, 20.0])}]]
    app_logic.trajectoryRefocusTimes = [4.0, 6.0, 14.0, 16.0]
    app_logic.get_trajectory_excitation_times = lambda _time: np.array([1.0, 11.0])

    candidates = app_logic.find_spin_echo_candidates()

    assert candidates == [
        {"time": 7.0, "excitation_time": 1.0, "refocus_time": 4.0},
        {"time": 17.0, "excitation_time": 11.0, "refocus_time": 14.0},
    ]


def test_trajectory_zero_marker_is_added_to_derived_coherence_plots(app_logic: GUIapp) -> None:
    app_logic.trajectoryZeroReferenceTime = 1.25
    app_logic.channels = [
        [{"chanLabel": "Gradient Trajectory"}],
        [{"chanLabel": "Gradient Trajectory Residual"}],
        [{"chanLabel": "Coherence Order"}],
    ]
    app_logic.plots = [DummyCursorPlot(), DummyCursorPlot(), DummyCursorPlot()]

    app_logic.refresh_trajectory_flip_markers()

    for plot in app_logic.plots:
        assert (1.25, "Trajectory zero", "#7a3e00") in plot.markers


def test_trajectory_echo_detection_interpolates_between_gradient_samples(app_logic: GUIapp) -> None:
    app_logic.trajectoryZeroReferenceTime = 0.0
    app_logic.trajectoryRefocusTimes = [0.25]
    app_logic.channels = [[{
        "chanLabel": "Gradient Trajectory",
        "t": np.array([0.0, 1.0, 2.0]),
        "raw_data": np.array([0.0, 1.0, 0.0]),
        "data": np.array([0.0, -0.5, 0.5]),
    }]]

    candidates = app_logic.find_trajectory_echo_candidates()

    assert [candidate["time"] for candidate in candidates] == pytest.approx([1.5])
    assert candidates[-1]["residual"] == pytest.approx(0.0)


def test_trajectory_echo_detection_reports_a_local_minimum_not_every_low_sample(app_logic: GUIapp) -> None:
    app_logic.trajectoryZeroReferenceTime = 0.0
    app_logic.trajectoryRefocusTimes = [0.25]
    app_logic.trajectoryEchoRelativeTolerance = 0.05
    app_logic.channels = [[{
        "chanLabel": "Gradient Trajectory",
        "t": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "data": np.array([1.0, 0.03, 0.02, 0.03, 1.0]),
    }]]

    candidates = app_logic.find_trajectory_echo_candidates()

    assert [candidate["time"] for candidate in candidates] == pytest.approx([2.0])


def test_trajectory_residual_combines_all_gradient_axes(app_logic: GUIapp) -> None:
    time, residual = app_logic.compute_trajectory_residual([
        {"t": np.array([0.0, 1.0]), "data": np.array([3.0, 0.0])},
        {"t": np.array([0.0, 1.0]), "data": np.array([4.0, 0.0])},
    ])

    np.testing.assert_allclose(time, np.array([0.0, 1.0]))
    np.testing.assert_allclose(residual, np.array([5.0, 0.0]))


def test_trajectory_residual_includes_the_interpolated_trajectory_zero(app_logic: GUIapp) -> None:
    app_logic.trajectoryZeroReferenceTime = 0.5
    time, residual = app_logic.compute_trajectory_residual([
        {"t": np.array([0.0, 1.0, 2.0]), "data": np.array([-1.0, 1.0, 3.0])},
    ])

    np.testing.assert_allclose(time, np.array([0.0, 0.5, 1.0, 2.0]))
    np.testing.assert_allclose(residual, np.array([1.0, 0.0, 1.0, 3.0]))


def test_jump_to_next_and_previous_rf_pulse_target_selection(app_logic: GUIapp) -> None:
    app_logic.rfPulseStartTimes = np.array([1.0, 2.0, 4.0])
    app_logic.currentCursorTime = 1.5
    captured: list[float] = []
    app_logic.jump_to_rf_pulse_time = lambda target_time: captured.append(float(target_time))

    app_logic.jump_to_next_rf_pulse()
    app_logic.jump_to_previous_rf_pulse()

    np.testing.assert_allclose(np.array(captured), np.array([2.0, 1.0]))
