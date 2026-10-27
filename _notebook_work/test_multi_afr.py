"""Synthetic invariants for causal alignment and equal-trip AFR aggregation."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from afr_pipeline import AFRConfig, CHANNELS, align_trip, trip_cell_statistics, aggregate_cells


def raw_trip(seconds, target, measured, rpm=None, load=None):
    n = len(seconds)
    return pd.DataFrame({
        "Time": [str(pd.to_timedelta(s, unit="s")) for s in seconds],
        "RPM": rpm if rpm is not None else [1200.0] * n,
        "Load": load if load is not None else [400.0] * n,
        "TargetAFR": target,
        "LambdaSensor1": measured,
    })


def info(trip_id):
    return {"trip_id": trip_id, "log_date": "2018-07-13", "direction": "home2mimos"}


class MultiTripAFRChecks(unittest.TestCase):
    def setUp(self):
        self.config = AFRConfig(min_trip_samples=1, min_trip_updates=1, min_trips=2)

    def test_no_future_fill_and_channel_specific_freshness(self):
        raw = raw_trip([0.0, 0.2, 0.4], [np.nan, 1000.0, np.nan], [900.0, np.nan, np.nan])
        aligned, _, _, _ = align_trip(raw, self.config)
        at = lambda seconds: aligned.loc[pd.to_timedelta(seconds, unit="s")]
        self.assertTrue(pd.isna(at(0.1).TargetAFR), "Future target must not fill earlier timestamps")
        self.assertAlmostEqual(at(0.2).target_afr, 14.7)
        self.assertAlmostEqual(at(0.2).measured_afr, 13.23)
        self.assertTrue(pd.isna(at(0.3).measured_afr), "Sensor older than 0.25 s must expire")
        self.assertAlmostEqual(at(0.3).target_afr, 14.7, msg="Target can remain fresh after sensor expires")
        self.assertTrue(at(0.3).usable_target)
        self.assertFalse(at(0.3).usable_measured)
        self.assertFalse(at(0.3).usable_error)
        for channel in CHANNELS:
            valid = aligned[channel].notna()
            self.assertTrue((aligned.loc[valid, channel + "__source_s"]
                             <= aligned.loc[valid, "elapsed_s"] + 1e-9).all())

    def test_recording_gap_resets_fill_even_when_max_age_is_long(self):
        self.config.max_age_s["TargetAFR"] = 10.0
        raw = raw_trip([0.0, 0.2, 3.0, 3.2],
                       [1000.0, np.nan, np.nan, np.nan],
                       [900.0, 910.0, 950.0, 960.0])
        aligned, report, _, _ = align_trip(raw, self.config)
        self.assertEqual(report["segments"], 2)
        self.assertEqual(report["recording_breaks"], 1)
        self.assertTrue(pd.isna(aligned.loc[pd.to_timedelta(3.0, unit="s"), "TargetAFR"]))
        self.assertFalse(((aligned.elapsed_s > 0.2) & (aligned.elapsed_s < 3.0)).any())
        self.assertAlmostEqual(aligned.loc[pd.to_timedelta(3.0, unit="s"), "measured_afr"], 13.965)

    def test_last_nonnull_duplicate_timestamp_and_paired_error_sign(self):
        raw = raw_trip([0.0, 0.0, 0.1],
                       [1000.0, 990.0, 1000.0], [1100.0, np.nan, 900.0])
        aligned, report, channels, _ = align_trip(raw, self.config)
        first = aligned.iloc[0]
        self.assertEqual(report["extra_duplicate_time_rows"], 1)
        self.assertEqual(first.TargetAFR, 990.0)
        self.assertEqual(first.LambdaSensor1, 1100.0)
        self.assertAlmostEqual(first.afr_error, (1100.0 - 990.0) / 1000.0 * 14.7)
        self.assertGreater(first.afr_error, 0)
        self.assertAlmostEqual(aligned.iloc[1].afr_error, -1.47)
        conflicts = channels.set_index("channel").duplicate_conflict_timestamps
        self.assertEqual(conflicts["TargetAFR"], 1)

    def test_aggregation_equal_trip_weight_and_same_elapsed_sources(self):
        a = raw_trip([0.0, 0.1], [1000.0, 1000.0], [1100.0, 1100.0])
        b = raw_trip([0.0, 0.1], [900.0, 900.0], [1000.0, 1000.0])
        aligned_a, _, _, numeric_a = align_trip(a, self.config)
        aligned_b, _, _, numeric_b = align_trip(b, self.config)
        np.testing.assert_array_equal(aligned_a.TargetAFR__source_s, aligned_b.TargetAFR__source_s)
        cells = pd.concat([
            trip_cell_statistics(aligned_a, numeric_a, info("trip_A"), self.config),
            trip_cell_statistics(aligned_b, numeric_b, info("trip_B"), self.config),
        ], ignore_index=True)
        result = aggregate_cells(cells, self.config).set_index("metric")
        self.assertAlmostEqual(result.loc["target", "value"], (14.7 + 13.23) / 2)
        self.assertAlmostEqual(result.loc["measured", "value"], (16.17 + 14.7) / 2)
        self.assertAlmostEqual(result.loc["error", "value"], 1.47)
        self.assertTrue(result.supported.all())
        self.assertTrue(result.n_trips.eq(2).all())
        # Identical source elapsed times in different trips remain different observations.
        self.assertTrue(result.n_source_updates.eq(4).all())
        inflated = cells.copy()
        inflated.loc[inflated.trip_id.eq("trip_A"), "n_grid"] *= 1000
        inflated.loc[inflated.trip_id.eq("trip_A"), "represented_time_s"] *= 1000
        result_inflated = aggregate_cells(inflated, self.config).set_index("metric")
        pd.testing.assert_series_equal(result.value, result_inflated.value)
        self.assertTrue(result_inflated.n_grid.eq(2002).all())

    def test_target_and_sensor_coverage_are_independent(self):
        target_only = raw_trip([0.0, 0.1], [1000.0, 1000.0], [np.nan, np.nan])
        measured_only = raw_trip([0.0, 0.1], [np.nan, np.nan], [950.0, 950.0])
        a, _, _, numeric_a = align_trip(target_only, self.config)
        b, _, _, numeric_b = align_trip(measured_only, self.config)
        self.assertTrue(a.usable_target.all())
        self.assertFalse(a.usable_measured.any())
        self.assertFalse(a.usable_error.any())
        self.assertTrue(b.usable_measured.all())
        self.assertFalse(b.usable_target.any())
        self.assertFalse(b.usable_error.any())
        cells = pd.concat([
            trip_cell_statistics(a, numeric_a, info("target_only"), self.config),
            trip_cell_statistics(b, numeric_b, info("measured_only"), self.config),
        ], ignore_index=True)
        self.assertEqual(set(cells.metric), {"target", "measured"})
        result = aggregate_cells(cells, self.config)
        self.assertTrue(result.n_trips.eq(1).all())
        self.assertFalse(result.supported.any(), "Two different channel-only trips cannot create a supported paired cell")


if __name__ == "__main__":
    unittest.main(verbosity=2)
