from pathlib import Path

import numpy as np
import pytest

from simview_app.interaction import InteractionMixin
from utils import simUtilsBrkr


def test_press_schematic_selects_vapor_and_resolves_gradient_and_adc_durations(tmp_path: Path) -> None:
    (tmp_path / "_FCube1.output").write_text(
        'Delay list "VpD1" -- : 2 items\n==================\n  0.128256 0.062216\n\n'
        'Loopcounter list "avList" -- : 1 item\n==================\n  256\n',
    )
    (tmp_path / "pulseprogram.precomp").write_text(
        '# 10 "PRESS.ppg"\n'
        'start, 10u\n'
        'if (PVM_WsMode == 1) {\n'
        'd4 grad_ramp{0, 0, 80}\n'
        '} else {\n'
        'if (PVM_WsMode == 2) {\n'
        'VpD1[1] grad_ramp{Trim[0], Trim[1], 0}\n'
        '}\n'
        '}\n'
        'aq_job0 dwgen_job0\n'
        'NAVD4 dwgen_Navigator\n'
        'lo to start times avList\n',
    )
    (tmp_path / "method").write_text("##$PVM_WsMode=VAPOR\n##$PVM_SpecAcquisitionTime=466.944\n")
    (tmp_path / "acqp").write_text("##$Trim=( 2 )\n10 20\n##$NAVD4=0.031008\n")

    viewer = InteractionMixin()
    viewer.pulseProgramStructure = simUtilsBrkr.read_pulse_program_structure(str(tmp_path), "PRESS.ppg")
    events = viewer.build_full_ppg_schematic_events()

    gradients = [event for event in events if event["kind"] == "gradient"]
    assert len(gradients) == 1
    assert gradients[0]["duration"] == pytest.approx(0.062216)
    assert viewer.resolve_ppg_gradient_component("Trim[1]") == 20
    assert [event["duration"] for event in events if event["kind"] == "acquisition"] == pytest.approx(
        [0.466944, 0.031008],
    )
    assert next(event for event in events if event["kind"] == "loop")["count"] == 256


def test_schematic_retains_first_complete_recorded_gradient_for_compiler_line(tmp_path: Path) -> None:
    gcube = tmp_path / "_GCube.xml"
    gcube.write_text(
        '<pulseprogram timeunit="0.000001">\n'
        '<ev t="1" g1="4" g2="0" g3="0" ln="8" />\n'
        '<ev t="2" g2="-1" ln="12" />\n'
        '<ev t="10" cmd="ng" ln="8" />\n'
        '<ev t="20" g2="-1" ln="12" />\n'
        '<ev t="21" g2="-2" ln="12" />\n'
        '<ev t="40" g1="0" g2="0" g3="0" ln="8" />\n'
        '<ev t="50" g2="-3" ln="12" />\n'
        '<ev t="51" g2="-4" ln="12" />\n'
        '</pulseprogram>\n',
    )

    profiles = simUtilsBrkr.read_schematic_gradient_profiles(gcube, {12}, expected_samples=2)

    np.testing.assert_allclose(profiles[12][0], [0, 1e-6])
    np.testing.assert_array_equal(profiles[12][1], [[4, 4], [-1, -2], [0, 0]])


def test_ppg_output_fallback_preserves_repeated_source_lines(tmp_path: Path) -> None:
    (tmp_path / "_FCube1.output").write_text(
        "12 \"'WsOvs.mod': line 37---> OvsD0 grad_ramp{10, 0, 0}\"\n"
        "20 \"'WsOvs.mod': line 37---> OvsD0 grad_ramp{10, 0, 0}\n",
    )
    structure = simUtilsBrkr.read_pulse_program_structure(str(tmp_path), "PRESS.ppg")

    assert [record["internal_line"] for record in structure["records"]] == [12, 20]
