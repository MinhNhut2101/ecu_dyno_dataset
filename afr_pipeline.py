"""Xử lý nhiều nhật ký ECU và lập bảng AFR theo vùng, cân bằng giữa các chuyến."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import csv
import hashlib
import json
import math

import numpy as np
import pandas as pd


@dataclass
class AFRConfig:
    stoich_afr: float = 14.7
    raw_lambda_scale: float = 1000.0
    sample_period_s: float = 0.1
    segment_gap_s: float = 2.0
    max_age_s: dict = field(default_factory=lambda: {
        "RPM": 0.25, "Load": 0.25, "TargetAFR": 2.0, "LambdaSensor1": 0.25,
    })
    rpm_bin_width: int = 500
    load_bin_width: int = 50
    min_trip_samples: int = 30
    min_trip_updates: int = 5
    min_trips: int = 3
    target_spread_flag_afr: float = 0.3
    holdout_date_fraction: float = 0.2

    def validate(self):
        assert self.stoich_afr > 0 and self.raw_lambda_scale > 0
        assert self.sample_period_s > 0 and self.segment_gap_s > 0
        assert self.rpm_bin_width > 0 and self.load_bin_width > 0
        assert self.min_trip_samples >= 1 and self.min_trip_updates >= 1 and self.min_trips >= 1
        assert 0 < self.holdout_date_fraction < 1
        assert set(self.max_age_s) == {"RPM", "Load", "TargetAFR", "LambdaSensor1"}
        assert all(x >= 0 for x in self.max_age_s.values())


CHANNELS = ["RPM", "Load", "TargetAFR", "LambdaSensor1"]
METRICS = ["target", "measured", "error"]


def read_trip(path):
    """Đọc đúng cấu trúc của từng file; không giả định thứ tự cột giống nhau."""
    path = Path(path)
    data = path.read_bytes()
    lines = data.decode("utf-8-sig").splitlines()
    log_index = next(i for i, line in enumerate(lines) if line.startswith("Log :"))
    channels, types = [], {}
    for line in lines[:log_index]:
        if line.startswith("Channel :"):
            channels.append(line.split(" : ", 1)[1].strip())
        elif line.startswith("Type :") and channels:
            types[channels[-1]] = line.split(" : ", 1)[1].strip()
    columns = ["Time"] + channels
    if len(set(columns)) != len(columns):
        raise ValueError(f"{path.name}: tên kênh trùng nhau")
    missing = set(CHANNELS) - set(columns)
    if missing:
        raise ValueError(f"{path.name}: thiếu kênh bắt buộc {missing}")
    for row_number, row in enumerate(csv.reader(lines[log_index + 1:]), log_index + 2):
        if row and len(row) != len(columns):
            raise ValueError(f"{path.name}:{row_number}: số trường không khớp metadata")
    raw = pd.read_csv(path, skiprows=log_index + 1, names=columns)
    if raw.empty:
        raise ValueError(f"{path.name}: không có bản ghi")
    log_time = pd.to_datetime(lines[log_index].split(" : ", 1)[1], format="%Y%m%d %H:%M:%S")
    info = {
        "trip_id": path.stem, "file": path.name, "sha256": hashlib.sha256(data).hexdigest(),
        "log_datetime": str(log_time), "log_date": log_time.strftime("%Y-%m-%d"),
        "direction": path.stem.split("-", 1)[-1], "raw_rows": len(raw),
        "column_count": len(columns), "schema": "|".join(columns),
        "channel_types": json.dumps({c: types.get(c) for c in CHANNELS}, sort_keys=True),
    }
    return raw, info


def align_trip(raw, config):
    """Giữ giá trị quá khứ có giới hạn, độc lập cho từng đoạn và từng chuyến."""
    config.validate()
    time = pd.to_timedelta(raw.Time, errors="coerce")
    if time.isna().any() or time.diff().dropna().lt(pd.Timedelta(0)).any():
        raise ValueError("Time lỗi/đi lùi; cần kiểm tra hoặc xử lý qua nửa đêm trước")
    elapsed = time - time.iloc[0]
    numeric = raw[CHANNELS].apply(pd.to_numeric, errors="coerce")
    invalid = (raw[CHANNELS].notna() & numeric.isna()) | (numeric.notna() & ~np.isfinite(numeric))
    if invalid.any().any():
        raise ValueError(f"Số không hợp lệ: {invalid.sum().to_dict()}")
    source = numeric.copy()
    source["elapsed"] = elapsed
    grouped = source.groupby("elapsed", sort=True)
    conflicts = grouped[CHANNELS].nunique().gt(1).sum()
    events = grouped[CHANNELS].last()
    events["segment_id"] = (
        events.index.to_series().diff().dt.total_seconds().gt(config.segment_gap_s)
        .cumsum().to_numpy()
    )
    parts = []
    step = pd.to_timedelta(config.sample_period_s, unit="s")
    for segment_id, segment in events.groupby("segment_id", sort=True):
        grid = pd.timedelta_range(segment.index.min().ceil(step), segment.index.max().floor(step),
                                  freq=step, name="elapsed")
        if len(grid) == 0:
            continue
        union = segment.index.union(grid).sort_values()
        history = segment[CHANNELS].reindex(union).ffill()
        part = pd.DataFrame(index=grid)
        part["segment_id"] = segment_id
        now = pd.Series(grid.total_seconds(), index=grid)
        for channel in CHANNELS:
            recorded_at = pd.Series(segment.index.total_seconds(), index=segment.index).where(
                segment[channel].notna()
            )
            last = recorded_at.reindex(union).ffill().reindex(grid)
            age = now - last
            fresh = age.ge(-1e-9) & age.le(config.max_age_s[channel] + 1e-9)
            part[channel] = history[channel].reindex(grid).where(fresh)
            part[channel + "__age_s"] = age
            part[channel + "__source_s"] = last
            part[channel + "__carried"] = part[channel].notna() & age.gt(1e-9)
            valid = part[channel].notna()
            lookup = pd.Series(segment[channel].to_numpy(), index=segment.index.total_seconds())
            np.testing.assert_allclose(part.loc[valid, channel],
                                       lookup.reindex(last[valid].to_numpy()), rtol=0, atol=0)
            assert age[valid].ge(-1e-9).all() and age[valid].le(config.max_age_s[channel] + 1e-9).all()
        parts.append(part)
    if not parts:
        raise ValueError("Không có đoạn đủ dài cho lưới lấy mẫu")
    aligned = pd.concat(parts).sort_index()
    aligned["elapsed_s"] = aligned.index.total_seconds()
    aligned["target_lambda"] = aligned.TargetAFR / config.raw_lambda_scale
    aligned["measured_lambda"] = aligned.LambdaSensor1 / config.raw_lambda_scale
    aligned["target_afr"] = aligned.target_lambda * config.stoich_afr
    aligned["measured_afr"] = aligned.measured_lambda * config.stoich_afr
    position = aligned.RPM.gt(0) & aligned.Load.ge(0)
    aligned["usable_position"] = position
    aligned["usable_target"] = position & aligned.target_afr.gt(0)
    aligned["usable_measured"] = position & aligned.measured_afr.gt(0)
    aligned["usable_error"] = aligned.usable_target & aligned.usable_measured
    aligned["afr_error"] = (aligned.measured_afr - aligned.target_afr).where(aligned.usable_error)
    gaps = elapsed.diff().dt.total_seconds()
    report = {
        "unique_timestamps": len(events), "extra_duplicate_time_rows": int(time.duplicated().sum()),
        "fully_duplicate_rows": int(raw.duplicated().sum()),
        "duration_s": elapsed.iloc[-1].total_seconds(), "largest_gap_s": gaps.max(),
        "recording_breaks": int(gaps.gt(config.segment_gap_s).sum()),
        "segments": int(events.segment_id.nunique()), "grid_rows": len(aligned),
        "valid_position_rows": int(position.sum()),
        "valid_target_rows": int(aligned.usable_target.sum()),
        "valid_measured_rows": int(aligned.usable_measured.sum()),
        "valid_paired_rows": int(aligned.usable_error.sum()),
        "zero_sensor_raw_rows": int(numeric.LambdaSensor1.eq(0).sum()),
        "target_raw_min": numeric.TargetAFR.min(), "target_raw_max": numeric.TargetAFR.max(),
        "rpm_max": numeric.RPM.max(), "load_max": numeric.Load.max(),
    }
    channel_report = []
    for channel in CHANNELS:
        channel_report.append({
            "channel": channel, "raw_missing_pct": numeric[channel].isna().mean() * 100,
            "grid_missing_pct": aligned[channel].isna().mean() * 100,
            "grid_carried_pct": aligned[channel + "__carried"].mean() * 100,
            "duplicate_conflict_timestamps": int(conflicts[channel]),
            "max_allowed_age_s": config.max_age_s[channel],
        })
    if "AFRDifference" in raw:
        check = raw[["TargetAFR", "LambdaSensor1", "AFRDifference"]].apply(pd.to_numeric, errors="coerce").dropna()
        report["difference_checked_rows"] = len(check)
        report["difference_mismatch_rows"] = int((~np.isclose(
            check.TargetAFR - check.LambdaSensor1, check.AFRDifference
        )).sum())
    return aligned, report, pd.DataFrame(channel_report), numeric


def trip_cell_statistics(aligned, numeric, info, config):
    """Một hàng cho mỗi chuyến × vùng × đại lượng, trước khi gộp giữa chuyến."""
    points = aligned.loc[aligned.usable_position].copy()
    points["rpm_bin"] = np.floor(points.RPM / config.rpm_bin_width).astype(int)
    points["load_bin"] = np.floor(points.Load / config.load_bin_width).astype(int)
    keys = ["load_bin", "rpm_bin"]
    frames = []
    for metric, channel, raw_channel in [
        ("target", "target_afr", "TargetAFR"),
        ("measured", "measured_afr", "LambdaSensor1"),
        ("error", "afr_error", None),
    ]:
        selected = points.loc[points["usable_" + metric]].copy()
        if selected.empty:
            continue
        if metric == "error":
            first, second = "TargetAFR__source_s", "LambdaSensor1__source_s"
        else:
            first = second = raw_channel + "__source_s"
        stats = selected.groupby(keys).agg(
            n_grid=(channel, "size"), median=(channel, "median"),
            q25=(channel, lambda x: x.quantile(.25)), q75=(channel, lambda x: x.quantile(.75)),
            updates_a=(first, "nunique"), updates_b=(second, "nunique"),
        )
        stats["n_updates"] = stats[["updates_a", "updates_b"]].min(axis=1)
        stats["eligible"] = stats.n_grid.ge(config.min_trip_samples) & stats.n_updates.ge(config.min_trip_updates)
        stats["represented_time_s"] = stats.n_grid * config.sample_period_s
        stats["within_trip_iqr"] = stats.q75 - stats.q25
        # So sánh với bản ghi gốc; không dùng ffill ở bảng đối chiếu này.
        direct = numeric.copy()
        if metric == "error":
            direct["value"] = (direct.LambdaSensor1 - direct.TargetAFR) / config.raw_lambda_scale * config.stoich_afr
            valid_raw = direct.LambdaSensor1.gt(0) & direct.TargetAFR.gt(0)
        else:
            direct["value"] = direct[raw_channel] / config.raw_lambda_scale * config.stoich_afr
            valid_raw = direct[raw_channel].gt(0)
        direct = direct.loc[direct.RPM.gt(0) & direct.Load.ge(0) & valid_raw].dropna(subset=["value"])
        direct["rpm_bin"] = np.floor(direct.RPM / config.rpm_bin_width).astype(int)
        direct["load_bin"] = np.floor(direct.Load / config.load_bin_width).astype(int)
        baseline = direct.groupby(keys)["value"].agg(raw_count="size", raw_median="median")
        stats = stats.join(baseline)
        stats["aligned_minus_raw_median"] = stats["median"] - stats.raw_median
        stats = stats.reset_index()
        stats["metric"] = metric
        for field_name in ["trip_id", "log_date", "direction"]:
            stats[field_name] = info[field_name]
        frames.append(stats)
    if not frames:
        raise ValueError("Chuyến không có mẫu AFR hợp lệ")
    return pd.concat(frames, ignore_index=True)


def aggregate_cells(trip_cells, config):
    """Median-of-trip-medians: mỗi chuyến đủ điều kiện có một phiếu cho mỗi ô."""
    keys = ["metric", "load_bin", "rpm_bin"]
    seen = trip_cells.groupby(keys).agg(
        n_trips_observed=("trip_id", "nunique"), all_observed_trip_median=("median", "median"),
        all_grid_samples=("n_grid", "sum"),
    )
    eligible = trip_cells.loc[trip_cells.eligible]
    qualified = eligible.groupby(keys).agg(
        value=("median", "median"), n_trips=("trip_id", "nunique"),
        n_dates=("log_date", "nunique"), n_grid=("n_grid", "sum"),
        n_source_updates=("n_updates", "sum"),
        trip_q25=("median", lambda x: x.quantile(.25)), trip_q75=("median", lambda x: x.quantile(.75)),
        trip_p10=("median", lambda x: x.quantile(.1)), trip_p90=("median", lambda x: x.quantile(.9)),
        trip_min=("median", "min"), trip_max=("median", "max"),
        raw_equal_trip_median=("raw_median", "median"),
    )
    summary = seen.join(qualified).reset_index()
    for name in ["n_trips", "n_dates", "n_grid", "n_source_updates"]:
        summary[name] = summary[name].fillna(0).astype(int)
    summary["supported"] = summary.n_trips.ge(config.min_trips)
    summary["trip_iqr"] = summary.trip_q75 - summary.trip_q25
    summary["trip_p90_p10"] = summary.trip_p90 - summary.trip_p10
    summary["target_variable"] = (
        summary.metric.eq("target") & summary.supported
        & summary.trip_p90_p10.gt(config.target_spread_flag_afr)
    )
    summary["aligned_minus_raw_equal_trip"] = summary.value - summary.raw_equal_trip_median
    return summary


def matrix_for(summary, metric, field, rpm_bins, load_bins, supported_only=False):
    chosen = summary.loc[summary.metric.eq(metric)].copy()
    if supported_only:
        chosen[field] = chosen[field].where(chosen.supported)
    return chosen.pivot(index="load_bin", columns="rpm_bin", values=field).reindex(
        index=load_bins, columns=rpm_bins
    )


def save_matrix(matrix, path, config):
    """CSV dùng khoảng RPM/Load thực thay vì mã ô nội bộ."""
    output = matrix.copy()
    output.index = [f"[{b * config.load_bin_width:g}, {(b + 1) * config.load_bin_width:g})" for b in output.index]
    output.columns = [f"[{b * config.rpm_bin_width:g}, {(b + 1) * config.rpm_bin_width:g})" for b in output.columns]
    output.to_csv(path, index_label="Load_bin")


def validate_by_date(trip_cells, config):
    """Giữ các chuyến cùng ngày chung một phía; so sánh các ngày cuối với lịch sử."""
    dates = sorted(trip_cells.log_date.unique())
    count = min(len(dates) - 1, max(1, math.ceil(len(dates) * config.holdout_date_fraction)))
    if count < 1:
        raise ValueError("Cần ít nhất hai ngày để kiểm chứng")
    held_dates = dates[-count:]
    train = trip_cells.loc[~trip_cells.log_date.isin(held_dates)]
    test = trip_cells.loc[trip_cells.log_date.isin(held_dates) & trip_cells.eligible].copy()
    assert set(train.trip_id).isdisjoint(set(test.trip_id))
    train_map = aggregate_cells(train, config)
    prediction = train_map.loc[train_map.supported, ["metric", "load_bin", "rpm_bin", "value"]]
    evaluated = test.merge(prediction.rename(columns={"value": "prediction"}),
                           on=["metric", "load_bin", "rpm_bin"], how="left", validate="many_to_one")
    evaluated["residual"] = evaluated["median"] - evaluated.prediction
    evaluated["absolute_error"] = evaluated.residual.abs()
    covered = evaluated.loc[evaluated.prediction.notna()]
    per_trip = covered.groupby(["metric", "trip_id"]).agg(
        mae=("absolute_error", "mean"), bias=("residual", "mean"), covered_cells=("prediction", "size")
    ).reset_index()
    rows = []
    for metric in METRICS:
        selected = evaluated.loc[evaluated.metric.eq(metric)]
        per_metric_trip = per_trip.loc[per_trip.metric.eq(metric)]
        rows.append({
            "metric": metric, "train_trips": train.trip_id.nunique(),
            "heldout_trips": test.trip_id.nunique(), "heldout_dates": len(held_dates),
            "eligible_test_trip_cells": len(selected), "predicted_trip_cells": selected.prediction.notna().sum(),
            "coverage_pct": 100 * selected.prediction.notna().mean(),
            "equal_trip_mae_afr": per_metric_trip.mae.mean(),
            "equal_trip_bias_afr": per_metric_trip.bias.mean(),
            "test_trips_with_predictions": len(per_metric_trip),
        })
    return pd.DataFrame(rows), evaluated, per_trip, held_dates


def run_all_trips(data_dir, output_dir, config=None, progress=print):
    config = config or AFRConfig()
    config.validate()
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise ValueError("Không tìm thấy CSV")
    output_dir = Path(output_dir)
    aligned_dir = output_dir / "aligned_trips"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    trip_reports, channel_reports, all_cells = [], [], []
    expected_types = None
    for index, path in enumerate(files, 1):
        raw, info = read_trip(path)
        if expected_types is None:
            expected_types = info["channel_types"]
        if info["channel_types"] != expected_types:
            raise ValueError(f"{path.name}: Type của kênh khác với các chuyến trước")
        aligned, audit, channel_audit, numeric = align_trip(raw, config)
        aligned["trip_id"] = info["trip_id"]
        aligned["log_date"] = info["log_date"]
        aligned["direction"] = info["direction"]
        cells = trip_cell_statistics(aligned, numeric, info, config)
        assert cells.groupby("metric").n_grid.sum().get("target", 0) == aligned.usable_target.sum()
        assert cells.groupby("metric").n_grid.sum().get("measured", 0) == aligned.usable_measured.sum()
        assert cells.groupby("metric").n_grid.sum().get("error", 0) == aligned.usable_error.sum()
        all_cells.append(cells)
        trip_reports.append({**info, **audit})
        channel_audit["trip_id"] = info["trip_id"]
        channel_reports.append(channel_audit)
        aligned.to_csv(aligned_dir / f"{path.stem}.csv.gz", index_label="elapsed", compression="gzip")
        if progress is not None and (index % 5 == 0 or index == len(files)):
            progress(f"Đã xử lý {index}/{len(files)} chuyến")
    trips = pd.DataFrame(trip_reports).sort_values(["log_datetime", "trip_id"])
    channel_quality = pd.concat(channel_reports, ignore_index=True)
    trip_cells = pd.concat(all_cells, ignore_index=True)
    assert trip_cells[["trip_id", "metric", "load_bin", "rpm_bin"]].duplicated().sum() == 0
    summary = aggregate_cells(trip_cells, config)
    rpm_bins = np.arange(trip_cells.rpm_bin.min(), trip_cells.rpm_bin.max() + 1)
    load_bins = np.arange(trip_cells.load_bin.min(), trip_cells.load_bin.max() + 1)
    validation, heldout_cells, heldout_trips, held_dates = validate_by_date(trip_cells, config)
    maps = {}
    for metric in METRICS:
        maps[metric] = matrix_for(summary, metric, "value", rpm_bins, load_bins, True)
        assert maps[metric].notna().sum().sum() == summary.loc[summary.metric.eq(metric), "supported"].sum()
        save_matrix(maps[metric], output_dir / f"{metric}_afr_map.csv", config)
        for field in ["all_observed_trip_median", "n_trips", "n_trips_observed", "trip_iqr", "trip_p90_p10"]:
            save_matrix(matrix_for(summary, metric, field, rpm_bins, load_bins),
                        output_dir / f"{metric}_{field}.csv", config)
    # Nhóm ngày để khảo sát thay đổi mục tiêu; không khẳng định là phiên bản hiệu chỉnh ECU.
    periods = [
        ("20180713_20180720", "2018-07-13", "2018-07-20"),
        ("20180727_20180803", "2018-07-27", "2018-08-03"),
        ("20180806_20180904", "2018-08-06", "2018-09-04"),
    ]
    period_summaries = {}
    for label, start, end in periods:
        selection = trip_cells.loc[trip_cells.log_date.between(start, end)]
        if selection.empty:
            continue
        period_summary = aggregate_cells(selection, config)
        period_summary["period"] = label
        period_summaries[label] = period_summary
        for metric in METRICS:
            save_matrix(matrix_for(period_summary, metric, "value", rpm_bins, load_bins, True),
                        output_dir / f"{label}_{metric}_afr_map.csv", config)
    target_period_comparison = pd.concat([
        frame.loc[frame.metric.eq("target"), ["load_bin", "rpm_bin", "period", "value", "n_trips", "supported"]]
        for frame in period_summaries.values()
    ], ignore_index=True)
    trip_cells.to_csv(output_dir / "trip_cell_statistics.csv", index=False)
    trips.to_csv(output_dir / "trip_quality.csv", index=False)
    channel_quality.to_csv(output_dir / "channel_quality.csv", index=False)
    summary.to_csv(output_dir / "combined_cell_statistics.csv", index=False)
    target_period_comparison.to_csv(output_dir / "target_period_comparison.csv", index=False)
    validation.to_csv(output_dir / "validation_summary.csv", index=False)
    heldout_cells.to_csv(output_dir / "heldout_trip_cells.csv", index=False)
    heldout_trips.to_csv(output_dir / "heldout_trip_scores.csv", index=False)
    manifest = {
        "config": asdict(config), "file_count": len(files), "raw_rows": int(trips.raw_rows.sum()),
        "aggregation": "median of eligible per-trip medians; equal weight per trip per cell",
        "source_update_identity": "trip_id plus per-channel source elapsed timestamp",
        "holdout_dates": held_dates, "validation": "latest dates held out; then final map refit using all trips",
        "error_sign": "measured minus target; paired before per-trip aggregation",
        "conversion_assumption": "raw = lambda*1000, gasoline stoichiometric AFR 14.7; user-approved",
        "missing_value_assumption": "causal carry-forward with channel-specific maximum age; unverified export semantics",
        "periods": periods, "interpolation": "none; unsupported cells remain missing",
        "purpose": "historical AFR summaries, not verified fuel or target corrections",
    }
    (output_dir / "settings.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "config": config, "trips": trips, "channel_quality": channel_quality,
        "trip_cells": trip_cells, "summary": summary, "maps": maps,
        "rpm_bins": rpm_bins, "load_bins": load_bins, "validation": validation,
        "period_summaries": period_summaries, "period_comparison": target_period_comparison,
        "heldout_dates": held_dates, "output_dir": output_dir,
    }


def plot_tables(result):
    """Vẽ bảng AFR có số trong ô, cùng chiều Load với hình tham khảo."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    config = result["config"]
    rpm_bins, load_bins = result["rpm_bins"], result["load_bins"]
    output_dir = result["output_dir"]
    def draw(matrix, title, filename, palette="turbo", norm=None, flags=None):
        cmap = plt.get_cmap(palette).copy()
        cmap.set_bad("#e5e7eb")
        norm = norm or Normalize(10, 20, clip=True)
        fig, ax = plt.subplots(figsize=(max(14, len(rpm_bins) * .85), max(9, len(load_bins) * .45)),
                               constrained_layout=True)
        values = matrix.to_numpy(dtype=float)
        image = ax.imshow(np.ma.masked_invalid(values), origin="upper", aspect="auto",
                          interpolation="nearest", norm=norm, cmap=cmap)
        ax.set_xticks(range(len(rpm_bins)))
        ax.set_xticklabels([f"[{b * config.rpm_bin_width:g}, {(b + 1) * config.rpm_bin_width:g})"
                            for b in rpm_bins], rotation=45, ha="left", fontsize=8)
        ax.set_yticks(range(len(load_bins)))
        ax.set_yticklabels([f"[{b * config.load_bin_width:g}, {(b + 1) * config.load_bin_width:g})"
                            for b in load_bins], fontsize=8)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("RPM — khoảng [cận dưới, cận trên)")
        ax.set_ylabel("Load gốc — thấp ở trên, cao ở dưới")
        ax.set_title(title, pad=18)
        ax.set_xticks(np.arange(-.5, len(rpm_bins), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(load_bins), 1), minor=True)
        ax.grid(which="minor", color="#94a3b8", linewidth=.4)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row, col in np.ndindex(values.shape):
            value = values[row, col]
            if np.isfinite(value):
                red, green, blue, _ = cmap(norm(value))
                color = "black" if .2126 * red + .7152 * green + .0722 * blue > .5 else "white"
                star = "*" if flags is not None and bool(flags.iloc[row, col]) else ""
                text = f"{value:.1f}{star}"
            else:
                text, color = "—", "#64748b"
            ax.text(col, row, text, ha="center", va="center", color=color, fontsize=8)
        fig.colorbar(image, ax=ax, label="AFR" if palette == "turbo" else "Sai lệch AFR")
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        return fig
    flags = matrix_for(result["summary"], "target", "target_variable", rpm_bins, load_bins).eq(True)
    figures = {}
    figures["target"] = draw(result["maps"]["target"],
        "Target AFR — trung vị cân bằng giữa 50 chuyến (*: mục tiêu khác nhau giữa chuyến)",
        "target_afr_map.png", flags=flags)
    figures["measured"] = draw(result["maps"]["measured"],
        "AFR cảm biến — trung vị cân bằng giữa các chuyến", "measured_afr_map.png")
    values = np.abs(result["maps"]["error"].to_numpy(dtype=float))
    limit = max(.5, float(np.nanmax(values))) if np.isfinite(values).any() else .5
    figures["error"] = draw(result["maps"]["error"], "AFR cảm biến − mục tiêu (cặp cùng mốc)",
        "afr_error_map.png", "RdBu_r", TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    # Bảng hỗ trợ và biến thiên giữa chuyến quan trọng ngang với bảng giá trị.
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        counts = matrix_for(result["summary"], metric, "n_trips", rpm_bins, load_bins).fillna(0)
        image = ax.imshow(counts.to_numpy(), origin="upper", aspect="auto", cmap="viridis", vmin=0,
                          vmax=max(1, result["trips"].trip_id.nunique()))
        ax.set_title({"target": "Target AFR", "measured": "AFR cảm biến", "error": "Cặp cảm biến–mục tiêu"}[metric])
        ax.set_xticks(range(len(rpm_bins)))
        ax.set_xticklabels([str(int((b + .5) * config.rpm_bin_width)) for b in rpm_bins], rotation=90, fontsize=8)
        ax.set_yticks(range(len(load_bins)))
        ax.set_yticklabels([str(int((b + .5) * config.load_bin_width)) for b in load_bins], fontsize=8)
        ax.set(xlabel="RPM — tâm vùng", ylabel="Load gốc — tâm vùng")
        fig.colorbar(image, ax=ax, label="Số chuyến đủ dữ liệu trong ô")
    fig.savefig(output_dir / "trip_coverage.png", dpi=150, bbox_inches="tight")
    figures["coverage"] = fig
    for period, summary in result["period_summaries"].items():
        matrix = matrix_for(summary, "target", "value", rpm_bins, load_bins, True)
        figures[period] = draw(matrix, f"Target AFR theo giai đoạn {period}", f"{period}_target_afr_map.png")
    return figures
