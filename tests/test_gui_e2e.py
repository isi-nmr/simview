import os
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pytest
from PyQt6 import QtCore, QtGui, QtTest, QtWidgets
from PyQt6.QtCore import QSettings

from simView import GUIapp
from simview_app.interaction import SchematicViewBox

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


def wait_for_bruker_load(gui: GUIapp, qapp: QtWidgets.QApplication, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        timeline = getattr(gui, "pulseProgramTimeline", None)
        plots = getattr(gui, "plots", [])
        if timeline is not None and plots:
            times, line_numbers = timeline
            if times is not None and line_numbers is not None and len(times) > 0 and len(line_numbers) > 0:
                return
        time.sleep(0.01)
    pytest.fail("Timed out waiting for asynchronous Bruker load to finish.")


@pytest.mark.parametrize("view", ["full_ppg", "executed"])
def test_schematic_holds_gradient_until_next_sample(qapp: QtWidgets.QApplication, view: str) -> None:
    gui = GUIapp.__new__(GUIapp)
    gui.rfPulseCalibrations = []
    gui.channels = [[{
        "key": "Gx",
        "type": "grads",
        "t": np.array([0.0, 2.0, 5.0, 7.0, 9.0]),
        "raw_data": np.array([40.0, -30.0, 20.0, 100.0, 0.0]),
    }], [{
        "key": "Gy",
        "type": "grads",
        "t": np.array([0.0, 2.0, 5.0]),
        "raw_data": np.array([0.0, 5.0, 0.0]),
    }]]
    plot = pg.PlotWidget()
    if view == "full_ppg":
        gui.render_ppg_epi_readout(plot, {
            "start": 1.0, "duration": 6.0, "source": "epi.mod", "line": 16, "acquisition_windows": [],
        }, 0.0, 6.0)
        scale = 0.42 / 100.0
    else:
        gui.render_sequence_schematic(plot, {"start": 1.0, "end": 7.0, "modules": []})
        scale = 0.38 / 40.0

    gx, gy = plot.listDataItems()[-2:]
    assert np.max(gy.getData()[1]) > 2.0
    gradient_curve = gx.curve
    path = gradient_curve.getPath()
    vertices = np.array([(path.elementAt(i).x, path.elementAt(i).y) for i in range(path.elementCount())])
    segments = np.diff(vertices, axis=0)
    assert not np.any((segments[:, 0] != 0) & (segments[:, 1] != 0))
    horizontal = segments[:, 0] > 0
    np.testing.assert_allclose(vertices[:-1][horizontal, 0], [0.0, 1.0, 4.0])
    np.testing.assert_allclose(vertices[:-1][horizontal, 0] + segments[horizontal, 0], [1.0, 4.0, 6.0])
    np.testing.assert_allclose(vertices[:-1][horizontal, 1], 3.0 + scale * np.array([40.0, -30.0, 20.0]))


def test_schematic_has_zero_gradient_until_first_sample() -> None:
    time, values = GUIapp.clip_schematic_gradient(
        {"t": np.array([2.0, 5.0]), "raw_data": np.array([40.0, -30.0])}, 1.0, 4.0,
    )

    np.testing.assert_array_equal(time, [1.0, 2.0, 4.0])
    np.testing.assert_array_equal(values, [0.0, 40.0, 40.0])


def test_full_ppg_draws_nested_loop_counts_above_waveforms(qapp: QtWidgets.QApplication) -> None:
    gui = GUIapp.__new__(GUIapp)
    program = ["start, 10u", "slice, 20u", "lo to slice times NSLICES", "lo to start times NR"]
    gui.pulseProgramStructure = {
        "records": [{"source": "demo.ppg", "line": index, "text": text} for index, text in enumerate(program)],
        "parameters": {"NSLICES": 1, "NR": 3},
    }
    plot = pg.PlotWidget()

    gui.render_full_ppg_schematic(plot)

    labels = {
        item.toPlainText(): item.pos().y() for item in plot.items() if isinstance(item, pg.TextItem)
    }
    assert 4.0 < labels["slices ×1"] < labels["Repetitions ×3"] < plot.viewRange()[1][1]
    assert labels["Repetitions ×3"] - labels["slices ×1"] >= 0.55


def test_full_ppg_uses_caption_for_expanded_epi_loop(qapp: QtWidgets.QApplication) -> None:
    gui = GUIapp.__new__(GUIapp)
    gui.pulseProgramStructure = {
        "records": [{"source": "epi.mod", "line": i, "text": code} for i, code in enumerate([
            "start, (rde1 de1 adc_job0 ph0 syrec_job0) (1u 1u ph1:r):f_job0",
            "0u gc_control", "{", "EpiD4 grad_shape <xyzb0> {a}",
            "loop PVM_EpiEchoCounter", "{", "EpiD5 grad_shape <xyzb0> {b}", "}", "}",
            "EpiD2 dwgen_job0", "EpiD0 eoscnpzbwrif_job0", "lo to start times NR",
        ])],
        "gradient_vectors": [np.zeros((3, 2)), np.zeros((3, 2))],
        "parameters": {
            "PVM_EpiEchoCounter": 32, "NR": 3, "PVM_EpiGradDwellTime": 1.0, "PVM_EpiPlateau": .004,
            "PVM_EpiModuleTime": .1, "EpiD0": 1e-6, "EpiD1": 3e-6, "EpiD2": 75e-6, "EpiD3": 1e-6,
        },
    }
    plot = pg.PlotWidget()
    gui.render_full_ppg_schematic(plot)
    labels = [item.toPlainText() for item in plot.items() if isinstance(item, pg.TextItem)]
    assert "EPI readout · 100 µs · 32 echo pairs" in labels
    brackets = [curve.toolTip() for curve in plot.listDataItems() if "Count expression:" in curve.toolTip()]
    assert len(brackets) == 1 and "Count expression: NR" in brackets[0]
    plot.close()


@pytest.mark.parametrize("source", ["MGE.ppg", "RARE.ppg", "custom.ppg"])
def test_full_ppg_labels_echo_loops_consistently(qapp: QtWidgets.QApplication, source: str) -> None:
    gui = GUIapp.__new__(GUIapp)
    gui.pulseProgramStructure = {
        "records": [
            {"source": source, "line": 100, "text": "echo, 10u"},
            {"source": source, "line": 104, "text": "lo to echo times l10"},
        ],
        "loop_values": {"l10": 9},
    }
    plot = pg.PlotWidget()
    gui.render_full_ppg_schematic(plot)

    labels = {item.toPlainText() for item in plot.items() if isinstance(item, pg.TextItem)}
    assert "Echoes ×9" in labels
    plot.close()


def test_full_ppg_overlays_distinct_loop_gradient_amplitudes(qapp: QtWidgets.QApplication) -> None:
    gui = GUIapp.__new__(GUIapp)
    program = [
        "ssfp, 10u",
        "d4 grad_ramp{10, PVM_EncGenValues1*g3, 0}",
        "0u PVM_EncGenValues1.inc",
        "lo to ssfp times l4",
    ]
    gui.pulseProgramStructure = {
        "records": [{"source": "FISP.ppg", "line": i, "text": code} for i, code in enumerate(program)],
        "loop_values": {"l4": 100},
        "parameters": {
            "d4": 144e-6, "g3": 20, "PVM_EncGenValues1": [0.0, -1.0, 1.0] + [0.0] * 97,
        },
        "gradient_ramps": [[0., 1.]] * 3,
        "gradient_ramp_durations": [100e-6] * 3,
    }
    plot = pg.PlotWidget()
    gui.render_full_ppg_schematic(plot)

    def endpoints(axis):
        return [curve.getData()[1][-1] for curve in plot.listDataItems()
                if curve.toolTip().startswith(f"{axis} PPG expression:")
                and np.ptp(curve.getData()[1]) > 0]

    np.testing.assert_allclose(sorted(endpoints("Gy")), [2 - .42 * .2, 2 + .42 * .2])
    np.testing.assert_allclose(endpoints("Gx"), [3 + .42 * .1])
    assert any("ssfp ×100" == item.toPlainText() for item in plot.items() if isinstance(item, pg.TextItem))
    plot.close()


def test_full_ppg_shows_frequency_offsets_and_the_loop_range(
    qapp: QtWidgets.QApplication, tmp_path: Path,
) -> None:
    from utils.simUtilsBrkr import read_pulse_program_structure

    (tmp_path / "_FCube1.output").write_text(
        'Frequency list "rxFreq" -- misc list #0\n: 1 item\n=====\n0.000125\n\n'
        'Frequency list "sliceFreq" -- misc list #3\n: 3 items\n=====\n-0.002 0.003\n0.010\n'
    )
    (tmp_path / "pulseprogram.precomp").write_text(
        "10u rxFreq(receive):f1\n"
        "0u sliceFreq.res\n"
        "slice, 10u sliceFreq :f1\n"
        "1m\n"
        "0u sliceFreq.inc\n"
        "lo to slice times 2\n"
    )
    gui = GUIapp.__new__(GUIapp)
    gui.pulseProgramStructure = read_pulse_program_structure(str(tmp_path), "demo.ppg")
    plot = pg.PlotWidget()
    gui.render_full_ppg_schematic(plot)

    labels = {item.toPlainText(): item.toolTip() for item in plot.items() if isinstance(item, pg.TextItem)}
    assert "rxFreq" in labels["Rx f1 · +125 Hz"]
    assert "sliceFreq" in labels["Tx f1 · −2 to +3 kHz"]
    plot.close()


def test_full_ppg_draws_every_ramp_after_gradient_variants_return_to_zero(qapp: QtWidgets.QApplication) -> None:
    gui = GUIapp.__new__(GUIapp)
    program = ["d4 grad_ramp{0, first, 0}", "d4 grad_off", "d4 grad_ramp{0, second, 0}"]
    zero = np.array([0.0])
    targets = np.linspace(-0.2, 0.2, 9)
    gui.pulseProgramStructure = {
        "records": [{"source": "demo.ppg", "line": i, "text": code} for i, code in enumerate(program)],
        "parameters": {"d4": 144e-6},
        "ramped_gradient_targets": [(zero, np.array([-.1, 0, .1]), zero), None, (zero, targets, zero)],
        "gradient_ramps": [[0, .5, 1]] * 3,
        "gradient_ramp_durations": [110e-6] * 3,
    }
    plot = pg.PlotWidget()
    gui.render_full_ppg_schematic(plot)

    ramps = [curve.getData()[1] for curve in plot.listDataItems()
             if curve.toolTip().startswith("Gy PPG expression: second") and len(curve.getData()[0]) == 3]
    np.testing.assert_allclose([values[0] for values in ramps], 2)
    np.testing.assert_allclose(sorted(values[-1] for values in ramps), 2 + .42 * targets)
    plot.close()


def test_full_ppg_scales_idle_delays_and_keeps_their_duration_labels(qapp: QtWidgets.QApplication) -> None:
    gui = GUIapp.__new__(GUIapp)
    gui.pulseProgramStructure = {
        "records": [
            {"source": "demo.ppg", "line": 1, "text": "d0"},
            {"source": "demo.ppg", "line": 2, "text": "d1"},
        ],
        "parameters": {"d0": 0.8, "d1": 0.001},
    }
    plot = pg.PlotWidget()
    widths = {}
    for scale in ("proportional", "compressed", "uniform"):
        gui.render_full_ppg_schematic(plot, delay_scale=scale)
        labels = {item.toPlainText() for item in plot.items() if isinstance(item, pg.TextItem)}
        assert {"d0 · 800 ms", "d1 · 1 ms"} <= labels
        widths[scale] = [
            np.ptp(curve.getData()[0]) for curve in plot.listDataItems() if curve.toolTip().startswith("PPG delay:")
        ]
    assert widths["proportional"][0] / widths["proportional"][1] == pytest.approx(800)
    assert widths["compressed"][0] < widths["proportional"][0] / 10
    assert widths["uniform"][0] == pytest.approx(widths["uniform"][1])


@pytest.mark.parametrize("scale", ["proportional", "uniform"])
def test_full_ppg_gradient_ramp_continues_across_following_delays(qapp: QtWidgets.QApplication, scale: str) -> None:
    gui = GUIapp.__new__(GUIapp)
    program = ["5u grad_ramp{-20, 0, 0}", "10u", "d1x", "d4 grad_ramp{10, 0, 0}"]
    gui.pulseProgramStructure = {
        "records": [{"source": "FISP.ppg", "line": i, "text": code} for i, code in enumerate(program)],
        "delay_names": ["d1x"],
        "parameters": {"d1x": 574.5e-6, "d4": 144e-6},
        "gradient_ramps": [[0., .5, 1.]] * 3,
        "gradient_ramp_durations": [110e-6] * 3,
    }
    plot = pg.PlotWidget()
    gui.render_full_ppg_schematic(plot, delay_scale=scale)

    curves = plot.listDataItems()
    prephaser = next(curve for curve in curves if curve.toolTip().startswith("Gx PPG expression: -20"))
    delays = [curve.getData()[0] for curve in curves if curve.toolTip().startswith("PPG delay:")]
    transition_end = delays[1][0] + np.ptp(delays[1]) * 95 / 574.5
    assert prephaser.getData()[0][-1] == pytest.approx(transition_end)
    hold = next(curve for curve in curves if not curve.toolTip() and np.allclose(curve.getData()[1], 3 - .42 * .2))
    assert hold.getData()[0][0] == pytest.approx(transition_end)
    plot.close()


def test_full_ppg_wheel_zoom_changes_one_axis_at_a_time(qapp: QtWidgets.QApplication) -> None:
    view = SchematicViewBox()
    view.separate_zoom = True
    plot = pg.PlotWidget(viewBox=view)
    plot.resize(600, 400)
    plot.show()
    view.setRange(xRange=(0, 10), yRange=(0, 10), padding=0)
    qapp.processEvents()
    view.fit_range = view.viewRange()
    pos = plot.mapFromScene(view.sceneBoundingRect().center())
    viewport = plot.viewport()
    zoom_ratios = []
    for modifier, axis in (
        (QtCore.Qt.KeyboardModifier.NoModifier, 0),
        (QtCore.Qt.KeyboardModifier.ShiftModifier, 1),
    ):
        before = view.viewRange()
        event = QtGui.QWheelEvent(
            QtCore.QPointF(pos), QtCore.QPointF(viewport.mapToGlobal(pos)),
            QtCore.QPoint(), QtCore.QPoint(0, 120), QtCore.Qt.MouseButton.NoButton,
            modifier, QtCore.Qt.ScrollPhase.ScrollUpdate, False,
        )
        qapp.sendEvent(viewport, event)
        after = view.viewRange()
        assert np.ptp(after[axis]) < np.ptp(before[axis])
        np.testing.assert_allclose(after[1 - axis], before[1 - axis])
        zoom_ratios.append(np.ptp(after[axis]) / np.ptp(before[axis]))
    assert zoom_ratios[0] < zoom_ratios[1] < 1

    QtTest.QTest.mouseDClick(viewport, QtCore.Qt.MouseButton.LeftButton, pos=pos)
    QtTest.QTest.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=pos)
    np.testing.assert_allclose(view.viewRange(), view.fit_range)
    plot.close()


def test_full_ppg_labels_rf_and_gradient_durations(qapp: QtWidgets.QApplication) -> None:
    gui = GUIapp.__new__(GUIapp)
    gui.channels = []
    gui.rfPulseCalibrations = []
    gui.pulseProgramStructure = {
        "records": [
            {"source": "demo.ppg", "line": 1, "text": "d4 grad_ramp{0, 0, 10}"},
            {"source": "demo.ppg", "line": 2, "text": "(p0:sp0 ph0):f1"},
            {"source": "demo.ppg", "line": 3, "text": "(p1 ph1):f1"},
        ],
        "parameters": {"d4": 144e-6, "p0": 0.0021, "p1": 0.0017},
    }
    plot = pg.PlotWidget()
    gui.render_full_ppg_schematic(plot)

    labels = {item.toPlainText() for item in plot.items() if isinstance(item, pg.TextItem)}
    assert {"d4 · 144 µs", "p0 · 2.1 ms", "p1 · 1.7 ms"} <= labels


def test_full_ppg_shows_compiled_preparation_despite_saved_method_flags(
    qapp: QtWidgets.QApplication, tmp_path: Path,
) -> None:
    from utils.simUtilsBrkr import read_pulse_program_structure

    program = [
        "if (PVM_FovSatOnOff == On)", "{", '"Counter = 1"', "}",
        ";EXEC_begin FovSat FovSat()", "if(PVM_FovSatOnOff == On) {",
        '"fovWait = fovWait / 1000"',
        'if "Counter == 0"', "{", "20u", "}", "else", "{", "(fovPul:sp41):f1", "}",
        "fovWait", "}", ";EXEC_end FovSat",
        ";EXEC_begin FatSup FatSup()", "if(PVM_FatSupPerform == Yes) {", "(p0:sp40):f1", "}",
        ";EXEC_end FatSup",
    ]
    (tmp_path / "pulseprogram.precomp").write_text("\n".join(program))
    (tmp_path / "method").write_text("##$PVM_FovSatOnOff=Off\n##$PVM_FatSupPerform=No\n")
    (tmp_path / "acqp").touch()
    delays = " ".join(["0.000144"] + ["0"] * 255)
    (tmp_path / "_FCube1.output").write_text(
        f"delays [sec]: 256 items\n====\n{delays}\npulses [usec]: 256 items\n====\n"
        "1000\nloopcounters: 256 items\n====\n\n\ndelay increments\n"
        "fovWait: 2 0 0 1 in lines: 16\n"
        "fovPul: 4 0 1 1 in lines: 14\n",
    )
    gui = GUIapp.__new__(GUIapp)
    gui.channels = []
    gui.rfPulseCalibrations = []
    gui.pulseProgramStructure = read_pulse_program_structure(str(tmp_path), "demo.ppg")
    plot = pg.PlotWidget()

    gui.render_full_ppg_schematic(plot)

    labels = {item.toPlainText() for item in plot.items() if isinstance(item, pg.TextItem)}
    assert {"FOV saturation", "fovWait · 144 µs"} <= labels
    assert "Fat suppression" not in labels
    assert any(curve.toolTip().startswith("PPG RF pulse: fovPul") for curve in plot.listDataItems())
    plot.close()


@pytest.mark.parametrize("command, expression", [("grad_shape", "30"), ("grad_ramp", "PVM_SelIrSpoilAmp")])
def test_full_ppg_uses_compiled_ramped_points_for_unresolved_gradients(
    qapp: QtWidgets.QApplication, tmp_path: Path, command: str, expression: str,
) -> None:
    from utils.simUtilsBrkr import read_pulse_program_structure

    (tmp_path / "_FCube1.output").touch()
    (tmp_path / "pulseprogram.precomp").write_text(
        f"144u groff\n432u {command} {{0, 0, {expression}}}\n144u grad_off\n"
    )
    rows = ["Ramped 0.000, 0 1: 0.000000 0 0 0 0"] * 12
    rows[8] = "Ramped 0.000, 0 1: -0.296333 0 0 0 0"
    rows[10] = "Ramped 0.000, 0 1: -0.297000 0 0 0 0"
    (tmp_path / "_GCube.output").write_text(
        " n_gpoints = 2, n_gvecp = 1, n_gvec = 2, n_slices = 2\n"
        " digsgradchanmap = 0, 1, 2, 3\n"
        "Gcon: Gradient Points\n----\n" + "\n".join(rows) +
        "\nGcon: GradientVectorPoints\n----\n"
        "ramp 0: 1 items\n----\n0.5\n"
        "ramp 1: 1 items\n----\n0.5\n"
        "ramp 2: 1 items\n----\n0.5\n",
    )
    gui = GUIapp.__new__(GUIapp)
    gui.pulseProgramStructure = read_pulse_program_structure(str(tmp_path), "demo.ppg", include_recorded_gradients=False)
    plot = pg.PlotWidget()

    gui.render_full_ppg_schematic(plot)

    ramp, hold = [curve for curve in plot.listDataItems()
                  if curve.toolTip().startswith(f"Gz PPG expression: {expression}")]
    assert ramp.opts["pen"].style() == hold.opts["pen"].style() == QtCore.Qt.PenStyle.SolidLine
    np.testing.assert_allclose(ramp.getData()[1], 1 + .42 * np.array([0, -.296333 / 2, -.296333]))
    np.testing.assert_allclose(hold.getData()[1], 1 - .42 * .296333)
    assert "Compiled Gcon ramped point" in ramp.toolTip()
    plot.close()


def test_full_ppg_follows_gcon_function_and_modifier_references(
    qapp: QtWidgets.QApplication, tmp_path: Path,
) -> None:
    from utils.simUtilsBrkr import read_pulse_program_structure

    (tmp_path / "_FCube1.output").touch()
    (tmp_path / "pulseprogram.precomp").write_text("432u grad_ramp{0, 0, unknown*100}\n144u grad_off\n")
    (tmp_path / "_GCube.output").write_text(
        " n_gpoints = 2, n_gvecp = 0, n_gvec = 0, n_slices = 1\n"
        " digsgradchanmap = 0, 1, 2, 3\n"
        "Gcon: Gradient Points\n----\n"
        + "Ramped 0.000, 0 1: 0 0 0 0 0\n" * 4
        + "Ramped 0.000, 0 1: 0.1 7 1 0 0\n"
        "Ramped 0.000, 0 1: 0 0 0 0 0\n"
        "Gcon: Gradient Points to Function Relation\n----\n"
        " 7: 1 3 -0.5 9\n 9: 2 4 0.4 -1\n"
        "Gcon: Gradient Shape Values\n----\n"
        "shape 3: 3 items\n----\n-1 0 1\n"
        "shape 4: 1 item\n----\n0.5\n"
        "Gcon: Ramp Shape Values\n----\n"
        "ramp 0: 1 items\n----\n0.5\n"
        "ramp 1: 1 items\n----\n0.5\n"
        "ramp 2: 1 items\n----\n0.5\n"
    )
    gui = GUIapp.__new__(GUIapp)
    gui.pulseProgramStructure = read_pulse_program_structure(str(tmp_path), "demo.ppg", include_recorded_gradients=False)
    plot = pg.PlotWidget()
    gui.render_full_ppg_schematic(plot)

    curves = [curve for curve in plot.listDataItems() if curve.toolTip().startswith("Gz PPG expression: unknown")]
    assert curves and all(curve.opts["pen"].style() == QtCore.Qt.PenStyle.SolidLine for curve in curves)
    targets = sorted({round(float(curve.getData()[1][-1]), 8) for curve in curves})
    np.testing.assert_allclose(targets, 1 + .42 * np.array([0., .1, .2]))
    plot.close()


def test_full_ppg_decodes_sampled_gradient_points_and_their_adc_window(
    qapp: QtWidgets.QApplication, tmp_path: Path,
) -> None:
    from utils.simUtilsBrkr import read_pulse_program_structure

    (tmp_path / "method").write_text("##$PVM_SpatDimEnum=<2D>\n")
    (tmp_path / "acqp").touch()
    (tmp_path / "pulseprogram.precomp").write_text(
        'define delay SpirD0 = 12u\ndefine list<grad_shape> a={$GradShape}\n# 101 "Spiral.mod"\n'
        "SpirD0 dwgen_job0 grad_shape <xyzb0> {a(), b(), 0, 0}\n"
        "if (PVM_SpatDim == 3) {\n144u grad_ramp{0, 0, unknown}\n}\n"
    )
    (tmp_path / "_FCube1.output").write_text(
        " [9]: CPD for f1 on NCO1:\n cpd program: DWELL-PROGRAM\n"
        " fixdel(10.000000 usec) fcubeRtps[0]: set=DwenA, clr=none\n"
        " SpirD0 deccmd f1: CPDSNG prog 16 (CPDPRG17) via ptr 9 NEXTGRAD\n"
    )
    (tmp_path / "_GCube.output").write_text(
        " n_gpoints = 1, n_gvecp = 0, n_gvec = 0, n_slices = 1\n"
        " digsgradchanmap = 0, 1, 2, 3\n"
        "Gcon: Gradient Points\n----\n"
        "Shaped 0.000, 0 3: 0 0 0 0 1\n"
        "Shaped 0.000, 0 3: 0 0 0 1 1\n"
        "Shaped 0.000, 0 3: 0 0 0 2 0\n"
        "Gcon: Gradient Points to Function Relation\n----\n"
        " 0: 1 1 0.5 -1\n 1: 2 2 -0.25 -1\n"
        "Gcon: Gradient Shape Values\n----\n"
        "shape 1: 3 items\n----\n0.2 -0.4 0\n"
        "shape 2: 3 items\n----\n0 0.8 0\n"
    )
    gui = GUIapp.__new__(GUIapp)
    gui.pulseProgramStructure = read_pulse_program_structure(str(tmp_path), "demo.ppg", include_recorded_gradients=False)
    plot = pg.PlotWidget()
    gui.render_full_ppg_schematic(plot)

    curves = plot.listDataItems()
    gradient = next(curve for curve in curves if curve.toolTip().startswith("Gx compiled gradient"))
    assert gradient.opts["stepMode"] == "right"
    np.testing.assert_allclose(gradient.getData()[1], 3 + .42 * np.array([.1, -.2, 0, 0]))
    adc = next(curve for curve in curves if curve.toolTip() == "ADC · 10 µs")
    assert np.ptp(adc.getData()[0]) == pytest.approx(.2)
    assert not any("unknown" in curve.toolTip() for curve in curves)
    plot.close()


def test_full_ppg_decodes_a_sampled_shape_with_a_phase_encoding_list(
    qapp: QtWidgets.QApplication, tmp_path: Path,
) -> None:
    from utils.simUtilsBrkr import read_pulse_program_structure

    (tmp_path / "pulseprogram.precomp").write_text(
        "vd1 grad_off\n6u grad_shape{0, shape()*phase, 0}\ngrad_ramp{0, 0, 20}\n144u\n"
    )
    (tmp_path / "_FCube1.output").write_text('Delay list "vd1" -- : 1 item\n====\n0.005\n')
    (tmp_path / "_GCube.output").write_text(
        " n_gpoints = 3, n_slices = 1\n digsgradchanmap = 0, 1, 2, 3\n"
        "Gcon: Gradient Points\n----\n"
        + "".join("Ramped 0, 0 1: 0 0 0 0 0\n" f"Shaped 0, 0 3: 0 0 0 0 {int(axis == 1)}\n"
                  "Ramped 0, 0 1: 0 0 0 0 0\n" for axis in range(3))
        + "Gcon: Gradient Points to Function Relation\n----\n0: 1 1 1 1\n1: 2 2 1 -1\n"
        "Gcon: Gradient Shape Values\n----\nshape 1: 3 items\n----\n0 0.5 0\n"
        "shape 2: 2 items\n----\n-1 1\n"
        "Gcon: Ramp Shape Values\n----\n" + "".join(f"ramp {axis}: 1 items\n----\n0.5\n" for axis in range(3))
    )
    gui = GUIapp.__new__(GUIapp)
    gui.pulseProgramStructure = read_pulse_program_structure(str(tmp_path), "demo.ppg", include_recorded_gradients=False)
    plot = pg.PlotWidget()
    gui.render_full_ppg_schematic(plot)
    traces = [curve for curve in plot.listDataItems() if curve.toolTip().startswith("Gy compiled gradient")]
    assert len(traces) == 2
    np.testing.assert_allclose(sorted(curve.getData()[1][1] for curve in traces), [2 - .21, 2 + .21])
    assert any(isinstance(item, pg.TextItem) and item.toPlainText() == "vd1 · 5 ms" for item in plot.items())
    ramp = next(curve for curve in plot.listDataItems() if curve.toolTip().startswith("Gz PPG expression: 20"))
    assert ramp.opts["pen"].style() == QtCore.Qt.PenStyle.SolidLine
    assert np.ptp(ramp.getData()[0]) > 0
    plot.close()


@pytest.mark.parametrize("slice_alter", ["No", "Yes"])
def test_full_ppg_uses_only_active_slice_polarities(slice_alter: str) -> None:
    gui = GUIapp.__new__(GUIapp)
    program = [
        "slice, 10u grad_shape{0,0,shape()*plusminus}",
        "10u grad_ramp{g6,0,g5*plusminus}",
        "10u grad_shape{readout()*projection,0,shape()*plusminus}",
        "if(SliceAlter == Yes) {",
        "0u plusminus.inc",
        "lo to slice times 2",
        "}",
    ]
    shaped = np.zeros((3, 2, 3))
    shaped[2, :, 1] = [-20, 20]
    mixed = shaped.copy()
    mixed[0, :, 1] = [-10, 10]
    ramped = (np.array([.2]), np.array([0.]), np.array([.3, -.3]))
    gui.pulseProgramStructure = {
        "records": [{"source": "demo.ppg", "line": i, "text": code} for i, code in enumerate(program)],
        "parameters": {"SliceAlter": slice_alter},
        "ramped_gradient_targets": [None, ramped, None],
        "sampled_gradient_points": {0: shaped, 2: mixed},
    }

    events = gui.build_full_ppg_schematic_events()
    shapes, ramp, projections = [event for event in events if event["kind"] == "gradient"]
    assert any(event["kind"] == "loop" for event in events) == (slice_alter == "Yes")
    if slice_alter == "Yes":
        np.testing.assert_array_equal(shapes["compiled_gradient_samples"], shaped)
        np.testing.assert_array_equal(ramp["compiled_ramped_target"][2], [.3, -.3])
        np.testing.assert_array_equal(projections["compiled_gradient_samples"], mixed)
    else:
        np.testing.assert_array_equal(shapes["compiled_gradient_samples"], shaped[:, 0])
        np.testing.assert_array_equal(ramp["compiled_ramped_target"][2], [.3])
        np.testing.assert_array_equal(projections["compiled_gradient_samples"][0], mixed[0])
        np.testing.assert_array_equal(projections["compiled_gradient_samples"][2, :, 1], [-20, -20])
        assert shaped[2, 1, 1] == mixed[2, 1, 1] == 20


@pytest.mark.parametrize(("scales", "scale_cursor", "expected_scales"), [
    ((1, 1, 1, 1), 3, (1,)),
    ((.5, 1, 1.5, 2), 3, (.5, 1, 1.5, 2)),
    ((.5, 1, 2), 3, (.5, 1, 2)),
    ((.5, 2), 2, (2,)),
])
def test_full_ppg_decodes_chained_test_gradient_and_paired_adc(
    qapp: QtWidgets.QApplication, tmp_path: Path, scales: tuple[float, ...], scale_cursor: int,
    expected_scales: tuple[float, ...],
) -> None:
    from utils.simUtilsBrkr import read_pulse_program_structure

    (tmp_path / "method").write_text("##$PVM_DummyScans=0\n")
    (tmp_path / "acqp").touch()
    (tmp_path / "pulseprogram.precomp").write_text(
        "define loopcounter lds = {$PVM_DummyScans}\n"
        "define delay readDur = 12u\n"
        "start, (rde1 de1 adc_job0 ph1) (1u 1u TxPhaseList:r):f_job0\n"
        "readDur dwgen_job0 grad_shape{0,0,shape()*enabled*scaling}\n"
        "1u enabled.inc\n"
        '"lds = lds - 1"\n'
        'if "lds> = 0" goto start\n'
    )
    (tmp_path / "_FCube1.output").write_text(
        " [9]: CPD for f1 on NCO1:\n cpd program: DWELL-PROGRAM\n"
        " fixdel(12.000000 usec) fcubeRtps[0]: set=DwenA, clr=none\n"
        " readDur deccmd f1: CPDSNG prog 16 (CPDPRG17) via ptr 9 NEXTGRAD\n"
    )
    (tmp_path / "_GCube.output").write_text(
        " n_gpoints = 1, n_slices = 1\n digsgradchanmap = 0, 1, 2, 3\n"
        "Gcon: Gradient Points\n----\n"
        + "Shaped 0, 0 3: 0 0 0 0 0\n" * 2
        + "Shaped 0, 0 3: 0 0 0 0 1\n"
        "Gcon: Gradient Points to Function Relation\n----\n"
        f"0: 1 1 1 1\n1: 2 2 1 2\n2: {scale_cursor} 3 0.5 -1\n"
        "Gcon: Gradient Shape Values\n----\n"
        "shape 1: 3 items\n----\n0 0.4 -0.2\n"
        "shape 2: 2 items\n----\n0 1\n"
        f"shape 3: {len(scales)} items\n----\n" + " ".join(map(str, scales)) + "\n"
    )
    gui = GUIapp.__new__(GUIapp)
    gui.pulseProgramStructure = read_pulse_program_structure(str(tmp_path), "demo.ppg", include_recorded_gradients=False)
    compiled = gui.pulseProgramStructure["sampled_gradient_points"][0]
    np.testing.assert_allclose(
        np.unique(compiled[2, :, 1]), [0, *(20 * scale for scale in expected_scales)],
    )
    plot = pg.PlotWidget()
    gui.render_full_ppg_schematic(plot)

    curves = plot.listDataItems()
    gradients = [curve for curve in curves if curve.toolTip().startswith("Gz compiled gradient")]
    assert len(gradients) == 1 + len(expected_scales)
    gradient = max(gradients, key=lambda curve: np.ptp(curve.getData()[1]))
    np.testing.assert_allclose(gradient.getData()[1], 1 + .42 * max(expected_scales) * np.array([0, .2, -.1, -.1]))
    adc, = [curve for curve in curves if curve.toolTip().startswith("ADC")]
    assert adc.toolTip() == "ADC · 12 µs"
    np.testing.assert_allclose(adc.getData()[0], gradient.getData()[0][[0, -1]])
    assert any(isinstance(item, pg.TextItem) and item.toPlainText() == "Dummy scans ×0" for item in plot.items())
    plot.close()


def test_full_ppg_keeps_parallel_gradients_on_the_rf_and_adc_timeline(qapp: QtWidgets.QApplication) -> None:
    gui = GUIapp.__new__(GUIapp)
    gui.channels = []
    gui.rfPulseCalibrations = []
    program = [
        "10u gc_control", "{", "100u grad_shape{a(),0,0}", "}",
        "20u", "(p0 ph0):f1", "d7 dwgen_job0", "aq_job0", "1m",
    ]
    gui.pulseProgramStructure = {
        "records": [{"source": "demo.ppg", "line": i, "text": code} for i, code in enumerate(program)],
        "parameters": {"p0": 20e-6, "d7": 10e-6, "PVM_AcquisitionTime": .06},
        "ramped_gradient_targets": [None],
        "sampled_gradient_points": {0: np.array([[0, 20, 0], [0, 0, 0], [0, 0, 0]])},
        "adc_gate_durations": {"d7": 50e-6},
    }
    plot = pg.PlotWidget()
    gui.render_full_ppg_schematic(plot, delay_scale="proportional")

    curves = plot.listDataItems()
    gradient = next(c for c in curves if c.toolTip().startswith("Gx compiled gradient"))
    rf = next(c for c in curves if c.toolTip().startswith("PPG RF pulse: p0"))
    adc, = [c for c in curves if c.toolTip().startswith("ADC ·")]
    np.testing.assert_allclose(gradient.getData()[0][[0, -1]], [0, 2])
    np.testing.assert_allclose(rf.getData()[0][[0, -1]], [.6, 1])
    np.testing.assert_allclose(adc.getData()[0], [1, 2])
    assert adc.toolTip() == "ADC · 50 µs"
    plot.close()


def test_full_ppg_marks_placeholder_waveforms(qapp: QtWidgets.QApplication) -> None:
    gui = GUIapp.__new__(GUIapp)
    gui.channels = []
    gui.rfPulseCalibrations = []
    instructions = [
        "d4 grad_ramp{0, 0, 10}",
        "d4 grad_ramp{0, 0, unavailable}",
        "(p0:sp0 ph0):f1",
        "d4 grad_off",
        "aq_job0 dwgen_job0",
    ]
    gui.pulseProgramStructure = {
        "records": [{"source": "demo.ppg", "line": i, "text": text} for i, text in enumerate(instructions)],
        "parameters": {"d4": 144e-6, "p0": 0.0021},
        "gradient_ramps": [[0.0, 1.0]] * 3,
        "gradient_ramp_durations": [110e-6] * 3,
    }
    plot = pg.PlotWidget()
    gui.render_full_ppg_schematic(plot)

    curves = plot.listDataItems()
    unknown = [curve for curve in curves if curve.toolTip().startswith("Gz PPG expression: unavailable")]
    assert unknown and all(
        curve.opts["pen"].style() == QtCore.Qt.PenStyle.DashLine and "Amplitude unresolved" in curve.toolTip()
        and "82%" not in curve.toolTip()
        for curve in unknown
    )
    for prefix in ("PPG RF pulse:", "Gradient hold:", "ADC placeholder:"):
        placeholder = next(curve for curve in curves if curve.toolTip().startswith(prefix))
        assert placeholder.opts["pen"].style() == QtCore.Qt.PenStyle.DashLine
    resolved = next(curve for curve in curves if curve.toolTip().startswith("Gz PPG expression: 10"))
    assert resolved.opts["pen"].style() == QtCore.Qt.PenStyle.SolidLine


def test_full_ppg_uses_compiled_rf_envelope_and_phase_cycle(qapp: QtWidgets.QApplication) -> None:
    gui = GUIapp.__new__(GUIapp)
    gui.channels = []
    gui.rfPulseCalibrations = []
    gui.pulseProgramStructure = {
        "records": [{"source": "demo.ppg", "line": 12, "text": "(p0:sp12 ph1):f1"}],
        "parameters": {"p0": 0.002},
        "rf_shapes": {("sp12", 1): [{
            "duration": 0.002, "amplitude": np.array([0.25, 1.0]), "phase": np.array([180., 270.]),
        }]},
        "rf_phase_lists": {("ph1", 1): np.array([90., 180.])},
    }
    plot = pg.PlotWidget()
    gui.render_full_ppg_schematic(plot)

    envelope = next(curve for curve in plot.listDataItems() if "Compiled RF envelope" in curve.toolTip())
    phase = next(curve for curve in plot.listDataItems() if curve.toolTip().startswith("RF phase"))
    assert envelope.opts["pen"].style() == QtCore.Qt.PenStyle.SolidLine
    assert envelope.opts["stepMode"] == phase.opts["stepMode"] == "right"
    np.testing.assert_allclose(envelope.getData()[1], 5.0 + 0.68 * np.array([0.25, 1, 1]))
    np.testing.assert_allclose(phase.getData()[1], 4.0 + 0.75 * np.array([270, 0, 0]) / 360)
    assert "ph1 · 90°" in phase.toolTip()


def test_executed_ppg_displays_recorded_phase_during_rf(qapp: QtWidgets.QApplication) -> None:
    gui = GUIapp.__new__(GUIapp)
    gui.rfPulseCalibrations = []
    gui.channels = [[{
        "key": key, "type": "NCO", "ind": "1", "t": np.array([0., .001, .002, .003]),
        "data": np.array(values),
    }] for key, values in {"am": [0, 1, 1, 0], "p0": [0, 340, 180, 0], "p1": [0, 30, 30, 0]}.items()]
    plot = pg.PlotWidget()
    gui.render_sequence_schematic(plot, {"start": 0., "end": .004, "modules": []})

    phase = next(curve for curve in plot.listDataItems() if curve.toolTip().startswith("RF phase"))
    assert phase.opts["stepMode"] == "right"
    np.testing.assert_allclose(
        phase.getData()[1], 4.0 + 0.75 * np.array([np.nan, 10, 210, np.nan]) / 360, equal_nan=True,
    )


def test_full_ppg_matches_recorded_rf_by_compiler_line_without_registry(qapp: QtWidgets.QApplication) -> None:
    gui = GUIapp.__new__(GUIapp)
    gui.rfPulseCalibrations = []
    gui.pulseProgramTimeline = (np.array([0., .001, .002, .003, .004]), np.array([8, 8, 99, 99, 100]))
    gui.channels = [[{
        "key": key, "type": "NCO", "ind": "1", "t": gui.pulseProgramTimeline[0],
        "data": np.array(values),
    }] for key, values in {"am": [0.25, 1, 1, 0.5, 0], "p0": [90, 180, 270, 0, 0]}.items()]
    gui.pulseProgramStructure = {
        "records": [
            {"source": "demo.ppg", "line": 12, "text": "(p0:sp12 ph1):f1", "internal_line": line}
            for line in (8, 99)
        ],
        "parameters": {"p0": 0.002},
    }
    plot = pg.PlotWidget()
    gui.render_full_ppg_schematic(plot)

    envelopes = [curve for curve in plot.listDataItems() if "Recorded RF envelope" in curve.toolTip()]
    assert len(envelopes) == 2
    np.testing.assert_allclose(envelopes[0].getData()[1], 5.0 + 0.68 * np.array([0., 0.25, 1., 1.]))
    np.testing.assert_allclose(envelopes[1].getData()[1], 5.0 + 0.68 * np.array([0., 1., 0.5, 0.5]))


def test_full_ppg_dense_preparation_keeps_waveforms_visible(qapp: QtWidgets.QApplication) -> None:
    gui = GUIapp.__new__(GUIapp)
    instructions = ["d4 grad_ramp{0, 0, 10}"] * 25 + ["aq_job0 dwgen_job0"]
    gui.pulseProgramStructure = {
        "records": [{"source": "demo.ppg", "line": i, "text": code} for i, code in enumerate(instructions)],
        "parameters": {"d4": 144e-6, "PVM_SpecAcquisitionTime": 466.944},
    }
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    plot = pg.PlotWidget()
    scroll.setWidget(plot)
    scroll.resize(1100, 560)
    scroll.show()
    qapp.processEvents()

    gui.render_full_ppg_schematic(plot)
    qapp.processEvents()

    intervals = [curve for curve in plot.listDataItems() if curve.toolTip().startswith("PPG gradient: d4")]
    assert len(intervals) == 25
    labels = [item for item in plot.items() if isinstance(item, pg.TextItem) and item.toPlainText() == "d4 · 144 µs"]
    assert len(labels) == 25
    assert all(label.isVisible() for label in labels)
    assert len({label.pos().y() for label in labels}) > 4
    view = plot.getViewBox()
    view_rect = view.sceneBoundingRect()
    assert all(view_rect.contains(label.sceneBoundingRect()) for label in labels)
    assert all(
        not label.sceneBoundingRect().intersects(other.sceneBoundingRect())
        for index, label in enumerate(labels) for other in labels[index + 1:]
    )
    waveform_top = view.mapViewToScene(QtCore.QPointF(0, 5)).y()
    waveform_bottom = view.mapViewToScene(QtCore.QPointF(0, 0)).y()
    assert waveform_bottom - waveform_top >= 240
    assert scroll.verticalScrollBar().maximum() > 0
    scroll.close()


def test_full_ppg_timing_labels_use_fewer_rows_on_zoom(qapp: QtWidgets.QApplication) -> None:
    gui = GUIapp.__new__(GUIapp)
    instructions = [
        "d4 grad_ramp{0, 0, 10}",
        *["d6"] * 4,
        "10u",
        "d5 grad_ramp{0, 0, 10}",
        "aq_job0 dwgen_job0",
    ]
    gui.pulseProgramStructure = {
        "records": [{"source": "demo.ppg", "line": i, "text": code} for i, code in enumerate(instructions)],
        "parameters": {"d4": 144e-6, "d5": 66.533e-3, "d6": 20e-6, "PVM_SpecAcquisitionTime": 466.944},
    }
    plot = pg.PlotWidget()
    plot.resize(1100, 600)
    plot.show()
    qapp.processEvents()
    gui.render_full_ppg_schematic(plot)

    def visible_labels():
        return [
            item for item in plot.items()
            if isinstance(item, pg.TextItem) and item.isVisible()
        ]

    assert {"d5 · 66.533 ms", "10 µs"} <= {item.toPlainText() for item in visible_labels()}
    overview_bottom = plot.viewRange()[1][0]
    end = next(
        curve.getData()[0][-1] for curve in plot.listDataItems()
        if curve.toolTip().startswith("PPG delay: 10u")
    )
    plot.setXRange(0, end, padding=0)

    assert {"10 µs", "d4 · 144 µs"} <= {item.toPlainText() for item in visible_labels()}
    assert plot.viewRange()[1][0] > overview_bottom
    plot.close()


def test_sequence_schematic_opens_as_an_independent_window(
    qapp: QtWidgets.QApplication, isolated_settings: None,
) -> None:
    gui = GUIapp(data={"time": {"val": [0.0, 1.0], "units": "ms"}})
    structure = {
        "records": [{"source": "demo.ppg", "line": 1, "text": "d1"}],
        "parameters": {"d1": 0.001},
    }
    gui.pulseProgramStructure = structure

    def dismiss_modal() -> None:
        modal = qapp.activeModalWidget()
        if isinstance(modal, QtWidgets.QDialog):
            modal.reject()

    def visible_schematics() -> list[QtWidgets.QWidget]:
        return [
            widget for widget in qapp.topLevelWidgets()
            if widget.isVisible() and widget.windowTitle() == "Pulse Sequence Schematic"
        ]

    gui.show()
    QtCore.QTimer.singleShot(0, dismiss_modal)
    try:
        gui.show_sequence_schematic()
        qapp.processEvents()
        schematic, = visible_schematics()
        assert schematic.windowHandle().transientParent() is None
        assert qapp.activeModalWidget() is None
        main_position = gui.pos()
        target = schematic.pos() + QtCore.QPoint(20, 20)
        schematic.move(target)
        assert schematic.pos() == target
        assert gui.pos() == main_position
        gui.show_sequence_schematic()
        assert visible_schematics() == [schematic]
        gui.resetApp()
        assert not schematic.isVisible()
        gui.pulseProgramStructure = structure
        gui.show_sequence_schematic()
        reopened, = visible_schematics()
        assert reopened is not schematic
        gui.close()
        assert not reopened.isVisible()
    finally:
        gui.close()


def test_full_ppg_channel_controls_combine_and_hide_traces(
    qapp: QtWidgets.QApplication, isolated_settings: None,
) -> None:
    gui = GUIapp(data={"time": {"val": [0.0, 1.0], "units": "ms"}})
    gui.pulseProgramStructure = {
        "records": [
            {"source": "demo.ppg", "line": 1, "text": "(p0 ph0):f1"},
            {"source": "demo.ppg", "line": 2, "text": "d4 grad_ramp{20, -10, 15}"},
            {"source": "demo.ppg", "line": 3, "text": "aq_job0 dwgen_job0"},
        ],
        "parameters": {"p0": .001, "d4": 144e-6, "PVM_AcquisitionTime": 1},
        "gradient_ramps": [[0, 1]] * 3,
        "gradient_ramp_durations": [110e-6] * 3,
    }
    gui.show_sequence_schematic()
    dialog = gui._sequenceSchematicDialog
    plot = dialog.findChild(pg.PlotWidget)

    def baselines():
        return {
            item.toolTip()[:2]: float(item.getData()[1][0]) for item in reversed(plot.listDataItems())
            if " PPG expression:" in item.toolTip()
        }

    def excursions():
        return {
            axis: max(
                np.max(np.abs(item.getData()[1] - baseline))
                for item in plot.listDataItems() if item.toolTip().startswith(f"{axis} PPG expression:")
            )
            for axis, baseline in baselines().items()
        }

    try:
        assert baselines() == {"Gx": 3.0, "Gy": 2.0, "Gz": 1.0}
        controls = {button.text(): button for button in dialog.findChildren(QtWidgets.QCheckBox)}
        assert controls["Autoscale gradients"].isChecked()
        np.testing.assert_allclose(list(excursions().values()), [.42, .42, .42])
        legend = dialog.findChild(QtWidgets.QWidget, "schematicLegend")
        legend_labels = {label.text(): label for label in legend.findChildren(QtWidgets.QLabel) if label.text()}
        assert {"RF", "RF phase", "Gx", "Gy", "Gz", "ADC", "Frequency offset", "Placeholder"} <= legend_labels.keys()
        assert legend_labels["Gy"].isVisible()
        plot.setXRange(.4, 1.8, padding=0)
        controls["Combine gradients"].click()
        assert len(set(baselines().values())) == 1
        np.testing.assert_allclose([excursions()[axis] for axis in ("Gx", "Gy", "Gz")], [.42, .21, .315])
        controls["Autoscale gradients"].click()
        np.testing.assert_allclose([excursions()[axis] for axis in ("Gx", "Gy", "Gz")], [.084, .042, .063])
        np.testing.assert_allclose(plot.viewRange()[0], [.4, 1.8])
        controls["Gy"].click()
        assert set(baselines()) == {"Gx", "Gz"}
        assert not legend_labels["Gy"].isVisible()
        assert legend_labels["Gx"].isVisible() and legend_labels["Gz"].isVisible()
        controls["RF"].click()
        controls["ADC"].click()
        tooltips = [item.toolTip() for item in plot.listDataItems()]
        assert any(tip.startswith("RF phase ·") for tip in tooltips)
        assert not any(tip.startswith(("PPG RF pulse:", "ADC ·")) for tip in tooltips)
        assert not any(isinstance(item, pg.LinearRegionItem) for item in plot.items())
        controls["RF phase"].click()
        assert not any(item.toolTip().startswith("RF phase ·") for item in plot.listDataItems())
    finally:
        dialog.close()
        gui.close()


@pytest.mark.parametrize(
    "choice, compiled",
    [("Signal Viewer", True), ("Full PPG", True), ("Cancel", True), ("Full PPG", False)],
)
def test_paravision_launch_chooses_view_before_parsing(
    qapp: QtWidgets.QApplication, isolated_settings: None, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, choice: str, compiled: bool,
) -> None:
    import simView

    simulation = tmp_path / "mrScanSim"
    simulation.mkdir()
    if compiled:
        (simulation / "_FCube1.output").write_text("1 \"'demo.ppg': line 1---> 20u\"\n")
    signal_loads = []
    monkeypatch.setattr(GUIapp, "loadData", lambda gui: signal_loads.append(gui.dataPath))

    def reject_recorded_gradients(*args, **kwargs):
        pytest.fail("Full PPG startup tried to read the recorded gradient XML")

    monkeypatch.setattr("utils.simUtilsBrkr.read_schematic_gradient_profiles", reject_recorded_gradients)

    def select_view(label: str) -> None:
        prompt = qapp.activeModalWidget()
        if label == "Full PPG" and not compiled:
            QtCore.QTimer.singleShot(0, lambda: select_view("Signal Viewer"))
        next(button for button in prompt.buttons() if button.text() == label).click()

    QtCore.QTimer.singleShot(0, lambda: select_view(choice))
    gui = simView.open_startup_view(str(tmp_path), paravision=True)
    if choice == "Cancel":
        assert gui is None
        assert not signal_loads
        return

    try:
        if choice == "Signal Viewer" or not compiled:
            assert signal_loads == [str(tmp_path)]
            assert gui.isVisible()
        else:
            assert not signal_loads
            assert not gui.isVisible()
            assert gui.pulseProgramStructure["records"][0]["text"] == "20u"
            assert gui._sequenceSchematicDialog.isVisible()
    finally:
        gui.close()


def test_open_folder_uses_qt_widget_dialog(
    qapp: QtWidgets.QApplication, isolated_settings: None, tmp_path: Path,
) -> None:
    gui = GUIapp(data={"time": {"val": [0.0, 1.0], "units": "ms"}})
    gui.dataPath = str(tmp_path)
    observed_options = []

    def cancel_dialog() -> None:
        dialog = qapp.activeModalWidget()
        if isinstance(dialog, QtWidgets.QFileDialog):
            observed_options.append(dialog.options())
            dialog.reject()

    try:
        QtCore.QTimer.singleShot(0, cancel_dialog)
        gui.open_folder()

        assert len(observed_options) == 1
        options = observed_options[0]
        assert options & QtWidgets.QFileDialog.Option.DontUseNativeDialog
        assert options & QtWidgets.QFileDialog.Option.ShowDirsOnly
        assert gui.dataPath == str(tmp_path)
    finally:
        gui.close()


def test_jump_targets_available_for_bruker_fixture(
    qapp: QtWidgets.QApplication,
    isolated_settings: None,
    bruker_fixture_path: Path,
) -> None:
    gui = GUIapp(simPath=str(bruker_fixture_path))
    try:
        wait_for_bruker_load(gui, qapp)
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
        wait_for_bruker_load(gui, qapp)
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
