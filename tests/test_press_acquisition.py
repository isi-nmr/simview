from pathlib import Path

import pytest

from utils.simUtilsBrkr import (
    annotate_fcube_event_jobs,
    extract_sample_acquisition_window_details,
    extract_sample_acquisition_windows,
    read_all_fcube_event_infos,
)


def test_press_adc_uses_navigator_eos_when_dwell_end_is_missing(tmp_path: Path) -> None:
    (tmp_path / "_FCube4.output").write_text(
        """1362 "'acqdec.mod': line 32---> (rde1 de1 adc_job0 ph0 syrec_job0)
1379 "'PRESS.ppg': line 161---> 1m eoscnpzbwrif_job0"
1395 "'Navigator.mod': line 127---> (rde1 de1 adc_Navigator ph0 syrec_Navigator)
1396 "'Navigator.mod': line 128---> NAVD4 dwgen_Navigator"
1397 "'Navigator.mod': line 129---> NAVD5 eoscnpzbwrif_Navigator"
"""
    )
    (tmp_path / "_FCube4.xml").write_text(
        """<pulseprogram path="/source/PRESS.ppg" timeunit="0.000001">
<ev t="0" nco="3" ncoc="2" ln="1362" />
<ev t="10" nco="3" rgp="0--0x1" ln="16000000" />
<ev t="11" nco="3" rgp="0--0" ln="16000000" />
<ev t="466954" nco="3" ln="16000001" />
<ev t="467098" nco="0" wr="0" if="0" ln="1379" />
<ev t="470000" nco="3" ncoc="2" ln="1395" />
<ev t="470004" nco="3" rgp="0--0x1" ln="21000000" />
<ev t="470005" nco="3" rgp="0--0" ln="21000000" />
<ev t="480005" nco="3" wr="1" if="1" ln="600" />
<ev t="501012" nco="0" wr="1" if="1" ln="1397" />
</pulseprogram>"""
    )

    infos = read_all_fcube_event_infos(str(tmp_path))
    windows = extract_sample_acquisition_windows(infos)["3"]
    details = extract_sample_acquisition_window_details(infos)

    assert [end - start for start, end in windows] == pytest.approx([0.466944, 0.031008])
    assert [(detail["start"], detail["end"]) for detail in details] == windows
    assert [detail["job_type"] for detail in details] == ["job 16", "Navigator"]


def test_dwell_end_matches_its_own_program_when_another_job_starts_first() -> None:
    info = {"events": [
        {"t": 1.0, "nco": "3", "ln": "100"},
        {"t": 2.0, "nco": "3", "rgp": "0--0x1", "ln": "16000000"},
        {"t": 3.0, "nco": "3", "ln": "101"},
        {"t": 4.0, "nco": "3", "rgp": "0--0x1", "ln": "21000000"},
        {"t": 5.0, "nco": "3", "ln": "21000001"},
        {"t": 6.0, "nco": "3", "ln": "16000001"},
    ]}
    annotate_fcube_event_jobs(info, {"100": "0", "101": "Navigator"})

    assert extract_sample_acquisition_windows([info]) == {"3": [(2.0, 6.0), (4.0, 5.0)]}
    assert [event["job"] for event in info["events"][-2:]] == ["Navigator", "0"]


def test_dwell_end_takes_precedence_over_eos_in_another_fcube() -> None:
    start = {"t": 1.0, "nco": "3", "rgp": "0--0x1", "ln": "16000000", "job": "0"}
    receiver_info = {"events": [start, {"t": 2.0, "nco": "3", "ln": "16000001", "job": "0"}]}
    write_info = {"events": [start, {"t": 2.5, "wr": "0", "ln": "1379", "job": "0"}]}
    annotate_fcube_event_jobs(write_info, {"16000000": "0", "1379": "0"}, {"1379"})

    assert extract_sample_acquisition_windows([write_info, receiver_info]) == {"3": [(1.0, 2.0)]}
