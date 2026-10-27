# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""PSG (RemLogic) parsing, including the midnight rollover that every Newcastle night has."""
import pandas as pd
import pytest
from src.data.newcastle import parse_psg, participant_id, SLEEP_STAGES


def _write(tmp_path, rows, date="10/06/2013"):
    head = ["RemLogic Event Export", "Patient:", "Patient ID:", f"Recording Date:\t{date}", "",
            "Events Included:", "R", "W", "N1", "N2", "N3", "Unscored", "", "Scoring Session:",
            "Scorer Name:  Sleep Scoring Assistant", "", 
            "Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]"]
    body = [f"{st}\tN/A\t{t}\t{st}\t30" for st, t in rows]
    p = tmp_path / "psg.txt"
    p.write_text("\n".join(head + body), encoding="latin-1")
    return p


def test_onset_wakeup_and_midnight_rollover(tmp_path):
    rows = [("W", "23:58:00"), ("W", "23:58:30"), ("N1", "23:59:00"), ("N2", "23:59:30"),
            ("N2", "00:00:00"), ("W", "00:00:30"), ("R", "00:01:00"), ("W", "00:01:30")]
    r = parse_psg(_write(tmp_path, rows))
    assert r["onset"] == pd.Timestamp("2013-06-10 23:59:00")
    assert r["wakeup"] == pd.Timestamp("2013-06-11 00:01:30")      # end of the last R epoch
    assert r["spt_min"] == pytest.approx(2.5)
    assert r["tst_min"] == pytest.approx(2.0)                       # 4 sleep epochs x 30 s
    assert (r["epochs"].start.diff().dropna() > pd.Timedelta(0)).all()


def test_rejects_file_without_sleep_or_header(tmp_path):
    with pytest.raises(ValueError):
        parse_psg(_write(tmp_path, [("W", "22:00:00"), ("W", "22:00:30")]))
    bad = tmp_path / "bad.txt"
    bad.write_text("RemLogic Event Export\nRecording Date:\t10/06/2013\n", encoding="latin-1")
    with pytest.raises(ValueError):
        parse_psg(bad)


def test_date_is_day_first(tmp_path):
    r = parse_psg(_write(tmp_path, [("N2", "22:00:00")], date="06/10/2013"))
    assert r["recording_date"] == pd.Timestamp("2013-10-06")        # dd/mm/yyyy


def test_participant_id_from_filenames():
    assert participant_id("MECSLEEP01_left wrist_012870_2013-06-12 11-40-37.bin") == 1
    assert participant_id("mecsleep10_psg.txt") == 10
    assert participant_id("readme.txt") is None
    assert {"N1", "N2", "N3", "R"} <= SLEEP_STAGES
