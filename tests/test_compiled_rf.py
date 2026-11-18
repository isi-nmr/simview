from pathlib import Path

import numpy as np
import pytest

from simview_app.schematic_rf import compiled_rf_shape
from utils.simUtilsBrkr import read_pulse_program_structure


def test_compiled_rf_shape_respects_the_logical_transmit_channel() -> None:
    shape = {"channel": 2, "duration": 0.001}
    structure = {"rf_shapes": {("sp3", 1): [shape]}}

    assert compiled_rf_shape({"rf_shape": "sp3", "rf_channel": 1}, structure) is None
    assert compiled_rf_shape({"rf_shape": "sp3", "rf_channel": 2}, structure) == (1, shape)


def test_compiled_rf_registry_preserves_variants_and_converts_shape_phase(tmp_path: Path) -> None:
    (tmp_path / "_FCube1.output").write_text(
        """    [7]: Power for f2 on NCO3:
    [8]: Power for f2 on NCO3:
shape registry:
===============
shape list [4]:
---------------
   shape [0]: SP50 on NCO3, poino = 7, power = 0.1W, offsFreq = -0.00125
              dur = 21600.000 us, phasecoherent = 0, 4 slices, Normal
       amplitudes: 4 items
       ----------------------
       0 2

       4 1
       phases: 4 items
       ------------------
       0 16384 32768 56282
shape list [5]:
---------------
   shape [0]: SP50 on NCO3, poino = 8, power = 0.1W
              dur = 1800.000 us, phasecoherent = 0, 2 slices, Normal
       amplitudes: 2 items
       ----------------------
       8 2
1094 "'demo.ppg': line 180--->   (p1: rfc_pulse rfc_phase):f2"

p1 Shape Pulse SP50:f2 deccmd f2: shape pulse from list via ptr 5 phase(0,0)
"""
    )

    structure = read_pulse_program_structure(str(tmp_path), "demo.ppg")
    variants = structure["rf_shapes"][("sp50", 3)]

    assert [shape["duration"] for shape in variants] == pytest.approx([0.0216, 0.0018])
    assert [shape["channel"] for shape in variants] == [2, 2]
    assert [shape["shape_list"] for shape in variants] == [4, 5]
    np.testing.assert_allclose(variants[0]["amplitude"], [0, 0.5, 1, 0.25])
    np.testing.assert_allclose(variants[0]["phase"], [0, 90, 180, 309.166259765625])
    np.testing.assert_allclose(variants[1]["amplitude"], [1, 0.25])
    assert variants[1]["phase"] is None
    assert variants[0]["offset_hz"] == pytest.approx(-1250)
    listed = compiled_rf_shape({"internal_line": 1094, "rf_shape": "rfc_pulse", "rf_channel": 2}, structure)
    assert listed is not None and listed[0] == 3
    assert listed[1]["shape_list"] == 5
    assert listed[1]["schedule_size"] == 1


def test_compiled_rf_phase_lists_use_their_own_nco_and_angle_units(tmp_path: Path) -> None:
    (tmp_path / "_FCube1.output").write_text(
        """Phaselist #0 Ph1 on channel 2
==============================
incr = 90.0, NCO = 1, maxidx = 4, chan = 1

original list: 4 items
-----------------------
0.0 90.0

180.0 270.0

Phaselist #1 ph1 on channel 2
==============================
incr = 90.0, NCO = 3, maxidx = 1, chan = 1
original list: 1 item
-----------------------
45.0

Phaselist #2 ph2 on channel 2
==============================
incr = 90.0, NCO = 3, maxidx = 2, chan = 1
original list: 2 items
-----------------------
30.0
"""
    )

    shape_data = read_pulse_program_structure(str(tmp_path), "demo.ppg")
    phases = shape_data["rf_phase_lists"]

    assert set(phases) == {("ph1", 1), ("ph1", 3)}
    np.testing.assert_allclose(phases[("ph1", 1)], [0, 90, 180, 270])
    np.testing.assert_allclose(phases[("ph1", 3)], [45])


def test_malformed_rf_shape_does_not_invent_an_amplitude_or_phase(tmp_path: Path) -> None:
    (tmp_path / "_FCube1.output").write_text(
        """   shape [0]: SP1 on NCO1, poino = 7
              dur = 1500.000 us
       amplitudes: 2 items
       ----------------------
       0 1
       phases: 2 items
       ------------------
       0 1.25
   shape [1]: SP2 on NCO1, poino = 8
              dur = 1500.000 us
       amplitudes: 2 items
       ----------------------
       2
"""
    )

    shape_data = read_pulse_program_structure(str(tmp_path), "demo.ppg")["rf_shapes"]

    assert set(shape_data) == {("sp1", 1)}
    assert shape_data[("sp1", 1)][0]["phase"] is None
