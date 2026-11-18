from pathlib import Path

import numpy as np
import pytest

from simview_app.interaction import InteractionMixin
from utils import simUtilsBrkr


def test_compiled_epi_builds_the_gradient_train_without_xml(tmp_path: Path) -> None:
    (tmp_path / "_FCube1.output").write_text("")
    scales = (-0.2, 0.1, 0.1, 0.02)
    vectors = []
    for vector in range(1, len(scales) + 1):
        vectors.append(f"{vector}: gvec_idx=0\nVector: Gradient Points\n")
        vectors.extend(f"Shaped 0.000, 0 2: 0.000000 0 0 {(vector - 1) * 4 + axis} 1\n" for axis in range(4))
    relations = [
        f"{index * 4 + axis}: {index + 1} {index} {scale if axis == 0 else 0} -1\n"
        for index, scale in enumerate(scales)
        for axis in range(4)
    ]
    # The first ramp's Y shape is scaled by a second, constant shape.
    relations[5] = "5: 2 1 0.2 16\n"
    relations.append("16: 5 4 0.5 -1\n")
    (tmp_path / "_GCube.output").write_text(
        "Gcon: GradientVectorPoints\n---\n" + "".join(vectors)
        + "Gcon: GradientVectors\n---\n0: 0, 1, 2, 3,\n"
        + "Gcon: Gradient Points to Function Relation\n---\n" + "".join(relations)
        + "Gcon: Gradient Shape Values\n---\n"
        + "shape 0: 2 items\n---\n1 1\nshape 1: 2 items\n---\n-1 1\n"
        + "shape 2: 2 items\n---\n1 1\nshape 3: 2 items\n---\n1 0\n"
        + "shape 4: 1 item\n---\n0.5\n"
    )
    structure = simUtilsBrkr.read_pulse_program_structure(str(tmp_path), "demo.ppg")
    structure["parameters"].update(
        PVM_EpiGradDwellTime=1.0,
        PVM_EpiPlateau=0.004,
        PVM_EpiModuleTime=0.036,
        PVM_EpiEchoCounter=2,
        EpiD1=3e-6,
        EpiD2=11e-6,
        EpiD3=1e-6,
        EpiD0=1e-6,
    )
    lines = [
        "(rde1 de1 adc_job0 ph0 syrec_job0) (1u 1u ph1:r):f_job0",
        "0u gc_control", "{", "EpiD4 grad_shape < xyzb0 > {a}",
        "loop PVM_EpiEchoCounter", "{", "EpiD5 grad_shape < xyzb0 > {b}",
        "EpiD11 grad_shape < xyzb0 > {c}", "}", "EpiD4 grad_shape < xyzb0 > {d}", "}",
        "EpiD1", "EpiD2 dwgen_job0", "EpiD3", "EpiD0 eoscnpzbwrif_job0",
    ]

    waveform = simUtilsBrkr.build_compiled_epi_waveform(structure, [{"text": line} for line in lines])

    assert waveform is not None
    assert waveform["duration"] == pytest.approx(36e-6)
    assert waveform["acquisition_windows"][0] == pytest.approx((23e-6, 34e-6))
    assert waveform["loop_bounds"] == pytest.approx((22e-6, 34e-6))
    assert waveform["gradient_shape_durations"] == pytest.approx({
        "EpiD4": 2e-6, "EpiD5": 2e-6, "EpiD11": 4e-6,
    })
    np.testing.assert_allclose(
        waveform["gradient_time"] * 1e6, [0, 20, 21, 22, 23, 24, 26, 28, 29, 30, 32, 34, 35, 36],
    )
    np.testing.assert_allclose(
        waveform["gradient_values"][0], [0, -20, -20, -10, 10, 10, 10, -10, 10, 10, 10, 2, 0, 0],
    )
    np.testing.assert_allclose(
        waveform["gradient_values"][1], [0, 0, 0, -5, 5, 0, 0, -5, 5, 0, 0, 0, 0, 0],
    )
    assert not np.any(waveform["gradient_values"][2])

    gui = InteractionMixin()
    gui.pulseProgramStructure = structure
    structure["records"] = [
        {"source": "DwiHead.mod", "line": 165, "text": "144u grad_ramp{100*DwGR1,0,0}"},
        *({"source": "epi.mod", "line": index, "text": line} for index, line in enumerate(lines)),
        {"source": "DwiHead.mod", "line": 194, "text": "144u grad_ramp{100*DwGR2,0,0}"},
    ]
    zero = np.array([0.0])
    structure["ramped_gradient_targets"] = [
        (np.array([.15]), zero, zero), None, None, None, None, (np.array([-.2]), zero, zero),
    ]
    before, readout, loop, after = gui.build_full_ppg_schematic_events()
    np.testing.assert_array_equal(before["compiled_ramped_target"][0], [.15])
    np.testing.assert_array_equal(after["compiled_ramped_target"][0], [-.2])
    assert readout["kind"] == "epi_readout" and loop["kind"] == "loop"
    assert loop["expanded"] and loop["count"] == 2
    np.testing.assert_array_equal(readout["gradient_values"], waveform["gradient_values"])
