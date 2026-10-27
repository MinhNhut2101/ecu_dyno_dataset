"""Nội suy tuyến tính 2D và bảng ngoại suy bổ sung có truy vết từng ô."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from afr_pipeline import AFRConfig, matrix_for, save_matrix


class LinearAFRSurface:
    """Tọa độ tính theo số khoảng RPM/Load; không dùng giá trị vừa điền làm nguồn."""
    def __init__(self, points, values, neighbors=8):
        self.points = np.asarray(points, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.neighbors = int(neighbors)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("Cần hai tọa độ RPM và Load")
        if len(self.values) != len(self.points) or len(self.points) < 3:
            raise ValueError("Cần ít nhất 3 điểm nguồn")
        if not np.isfinite(self.points).all() or not np.isfinite(self.values).all():
            raise ValueError("Nguồn phải là số hữu hạn")
        if len(np.unique(self.points, axis=0)) != len(self.points):
            raise ValueError("Tọa độ nguồn bị trùng")
        if np.linalg.matrix_rank(np.c_[np.ones(len(self.points)), self.points]) < 3:
            raise ValueError("Không thể dựng mặt 2D từ các điểm nguồn thẳng hàng")
        if self.neighbors < 3:
            raise ValueError("Số láng giềng phải từ 3 trở lên")
        self.triangulation = Delaunay(self.points)
        self.linear = LinearNDInterpolator(self.triangulation, self.values, fill_value=np.nan)

    def predict(self, query):
        query = np.asarray(query, dtype=float).reshape(-1, 2)
        if not np.isfinite(query).all():
            raise ValueError("Tọa độ cần dự đoán phải hữu hạn")
        simplex = self.triangulation.find_simplex(query, tol=1e-10)
        strict = np.asarray(self.linear(query), dtype=float).reshape(-1)
        full = strict.copy()
        origins, distances, counts, ranks, conditions = [], [], [], [], []
        for index, point in enumerate(query):
            distance = np.linalg.norm(self.points - point, axis=1)
            nearest = int(np.argmin(distance))
            distances.append(distance[nearest])
            if distance[nearest] < 1e-10:
                # Giá trị nguồn được bảo toàn tuyệt đối, không chỉ gần bằng.
                strict[index] = full[index] = self.values[nearest]
                origins.append("supported")
                counts.append(1); ranks.append(0); conditions.append(np.nan)
            elif simplex[index] >= 0:
                if not np.isfinite(strict[index]):
                    raise ValueError("Lỗi số học nội suy trong miền nguồn")
                origins.append("linear_interpolation")
                counts.append(3); ranks.append(3); conditions.append(np.nan)
            else:
                # Nội suy thuần không có giá trị ngoài miền; chỉ full-map dùng mặt phẳng.
                strict[index] = np.nan
                order = np.argsort(distance, kind="stable")
                k = min(self.neighbors, len(self.points))
                while True:
                    chosen = order[:k]
                    design = np.c_[np.ones(k), self.points[chosen] - point]
                    weights = 1.0 / (1.0 + distance[chosen] ** 2)
                    weighted_design = design * np.sqrt(weights[:, None])
                    coefficients, _, rank, singular = np.linalg.lstsq(
                        weighted_design, self.values[chosen] * np.sqrt(weights), rcond=None
                    )
                    condition = singular[0] / singular[-1] if singular[-1] > 0 else np.inf
                    if rank == 3 and condition < 1e8:
                        break
                    if k == len(self.points):
                        raise ValueError("Không đủ hạng/điều kiện số để ngoại suy mặt phẳng 2D")
                    k += 1
                full[index] = coefficients[0]
                origins.append("linear_extrapolation")
                counts.append(k); ranks.append(rank); conditions.append(condition)
        assert np.isfinite(full).all()
        audit = pd.DataFrame({
            "origin": origins, "inside_source_hull": simplex >= 0,
            "nearest_anchor_distance_bins": distances, "neighbors_used": counts,
            "fit_rank": ranks, "fit_condition": conditions,
        })
        return strict, full, audit


def complete_matrix(matrix, neighbors=8):
    """Điền trên cùng lưới và giữ riêng bảng nội suy thuần với bảng đầy đủ."""
    x, y = np.meshgrid(matrix.columns.to_numpy(dtype=float), matrix.index.to_numpy(dtype=float))
    query = np.c_[x.ravel(), y.ravel()]
    flat = matrix.to_numpy(dtype=float).ravel()
    known = np.isfinite(flat)
    model = LinearAFRSurface(query[known], flat[known], neighbors)
    strict, full, audit = model.predict(query)
    np.testing.assert_array_equal(full[known], flat[known])
    np.testing.assert_array_equal(strict[known], flat[known])
    audit["rpm_bin"] = query[:, 0].astype(int)
    audit["load_bin"] = query[:, 1].astype(int)
    audit["source_value"] = flat
    audit["estimated_value"] = full
    audit["far_from_support"] = audit.nearest_anchor_distance_bins.gt(2)
    audit["outside_anchor_value_range"] = (full < flat[known].min() - 1e-10) | (full > flat[known].max() + 1e-10)
    audit["nonpositive_prediction"] = full <= 0
    make = lambda values: pd.DataFrame(values.reshape(matrix.shape), index=matrix.index, columns=matrix.columns)
    return {"source": matrix.copy(), "strict": make(strict), "full": make(full), "audit": audit}


def spatial_validation(matrix, neighbors=8):
    """Giữ riêng cả dải RPM hoặc dải Load; tách điểm nội suy và ngoại suy khi chấm."""
    x, y = np.meshgrid(matrix.columns.to_numpy(dtype=float), matrix.index.to_numpy(dtype=float))
    points = np.c_[x.ravel(), y.ravel()]
    values = matrix.to_numpy(dtype=float).ravel()
    finite = np.isfinite(values)
    points, values = points[finite], values[finite]
    records = []
    for axis, axis_name in [(0, "rpm_column"), (1, "load_row")]:
        for held in np.unique(points[:, axis]):
            test = points[:, axis] == held
            if (~test).sum() < 3:
                continue
            model = LinearAFRSurface(points[~test], values[~test], neighbors)
            _, prediction, audit = model.predict(points[test])
            block = audit.copy()
            block["split_axis"] = axis_name
            block["held_bin"] = int(held)
            block["rpm_bin"] = points[test, 0].astype(int)
            block["load_bin"] = points[test, 1].astype(int)
            block["actual"] = values[test]
            block["prediction"] = prediction
            block["error"] = prediction - values[test]
            block["absolute_error"] = block.error.abs()
            records.append(block)
    details = pd.concat(records, ignore_index=True)
    scores = details.groupby(["split_axis", "origin"]).agg(
        n_test_cells=("actual", "size"), mae_afr=("absolute_error", "mean"),
        bias_afr=("error", "mean"), max_absolute_error_afr=("absolute_error", "max"),
        max_test_distance_bins=("nearest_anchor_distance_bins", "max"),
    ).reset_index()
    return scores, details


def run_completion(source_dir="outputs/all_trips_afr", neighbors=8):
    source_dir = Path(source_dir)
    settings = json.loads((source_dir / "settings.json").read_text(encoding="utf-8"))
    config = AFRConfig(**settings["config"])
    summary = pd.read_csv(source_dir / "combined_cell_statistics.csv")
    # Dùng đúng toàn bộ lưới của phần 10, kể cả những hàng/cột đang trống.
    example = pd.read_csv(source_dir / "target_afr_map.csv", index_col=0)
    intervals_rpm = [pd.Interval(*map(float, label.strip("[)").split(",")), closed="left") for label in example.columns]
    intervals_load = [pd.Interval(*map(float, label.strip("[)").split(",")), closed="left") for label in example.index]
    rpm_bins = np.array([int(v.left / config.rpm_bin_width) for v in intervals_rpm])
    load_bins = np.array([int(v.left / config.load_bin_width) for v in intervals_load])
    output = source_dir / "linear_completion"
    output.mkdir(parents=True, exist_ok=True)
    data_sets = {"all_50_trips": summary}
    period = pd.read_csv(source_dir / "target_period_comparison.csv")
    for label, frame in period.groupby("period", sort=True):
        frame = frame.copy()
        frame["metric"] = "target"
        data_sets[label] = frame
    results, summaries, validations, validation_details = {}, [], [], []
    for dataset, frame in data_sets.items():
        for metric in (["target", "measured", "error"] if dataset == "all_50_trips" else ["target"]):
            source = matrix_for(frame, metric, "value", rpm_bins, load_bins, supported_only=True)
            completed = complete_matrix(source, neighbors)
            key = f"{dataset}_{metric}"
            results[key] = completed
            completed["metric"] = metric
            completed["dataset"] = dataset
            audit = completed["audit"]
            selected_stats = frame.loc[frame.metric.eq(metric), ["rpm_bin", "load_bin", "n_trips"]]
            audit = audit.merge(selected_stats.rename(columns={"n_trips": "source_cell_supporting_trips"}),
                                on=["rpm_bin", "load_bin"], how="left", validate="one_to_one")
            completed["audit"] = audit
            for name in ["source", "strict", "full"]:
                save_matrix(completed[name], output / f"{key}_{name}.csv", config)
            origin_matrix = pd.DataFrame(audit.origin.to_numpy().reshape(source.shape), index=source.index, columns=source.columns)
            save_matrix(origin_matrix, output / f"{key}_cell_origin.csv", config)
            audit.to_csv(output / f"{key}_cell_audit.csv", index=False)
            counts = audit.origin.value_counts()
            summaries.append({
                "dataset": dataset, "metric": metric, "total_cells": source.size,
                "supported_cells": int(counts.get("supported", 0)),
                "interpolated_cells": int(counts.get("linear_interpolation", 0)),
                "extrapolated_cells": int(counts.get("linear_extrapolation", 0)),
                "far_extrapolated_cells": int((audit.origin.eq("linear_extrapolation") & audit.far_from_support).sum()),
                "full_min": audit.estimated_value.min(), "full_max": audit.estimated_value.max(),
                "nonpositive_cells": int(audit.nonpositive_prediction.sum()),
                "outside_anchor_value_range_cells": int(audit.outside_anchor_value_range.sum()),
            })
            scores, details = spatial_validation(source, neighbors)
            scores["dataset"], scores["metric"] = dataset, metric
            details["dataset"], details["metric"] = dataset, metric
            validations.append(scores)
            validation_details.append(details)
    coverage = pd.DataFrame(summaries)
    validation = pd.concat(validations, ignore_index=True)
    coverage.to_csv(output / "completion_summary.csv", index=False)
    validation.to_csv(output / "spatial_validation.csv", index=False)
    pd.concat(validation_details, ignore_index=True).to_csv(output / "spatial_validation_details.csv", index=False)
    completion_settings = {
        "source_settings": str(source_dir / "settings.json"),
        "neighbors_initial": neighbors, "neighbors_rule": "expand until full rank3 and condition<1e8",
        "coordinate_scale": "one RPM bin and one Load bin each have unit distance",
        "interpolation": "piecewise linear barycentric interpolation inside Delaunay convex hull",
        "outside_hull": "separate exploratory map: centered local weighted least squares plane",
        "weights": "1/(1+distance_in_bin_units**2)",
        "anchors": "only supported cells from each dataset; never reuse generated values as anchors",
        "source_preservation": "all supported values unchanged exactly",
        "clipping": "none; range and nonpositive flags exported",
        "validation": "leave one supported RPM column out; separately leave one supported Load row out",
        "validation_limit": "near known support only; no evidence of far-region accuracy",
        "far_distance_threshold_bins": 2,
        "purpose": "exploratory table completion; extrapolation is not measured evidence or verified target selection",
        "references": [
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html",
            "https://docs.scipy.org/doc/scipy/tutorial/interpolate/extrapolation_examples.html",
        ],
    }
    (output / "settings.json").write_text(json.dumps(completion_settings, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"results": results, "coverage": coverage, "validation": validation,
            "output_dir": output, "config": config, "rpm_bins": rpm_bins, "load_bins": load_bins}


def plot_completion(completion):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    config = completion["config"]
    figures = {}
    origin_colors = ["#38bdf8", "#4ade80", "#fdba74"]
    for key, result in completion["results"].items():
        matrix = result["full"]
        values = matrix.to_numpy(dtype=float)
        origins = result["audit"].origin.to_numpy().reshape(values.shape)
        marker = {"supported": "", "linear_interpolation": "I", "linear_extrapolation": "E"}
        if result["metric"] == "error":
            from matplotlib.colors import TwoSlopeNorm
            limit = max(.5, float(np.abs(values).max()))
            norm, palette = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), "RdBu_r"
        else:
            norm = Normalize(vmin=min(10, float(values.min())), vmax=max(20, float(values.max())))
            palette = "turbo"
        cmap = plt.get_cmap(palette)
        fig, ax = plt.subplots(figsize=(15, 11), constrained_layout=True)
        image = ax.imshow(values, aspect="auto", origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels([f"[{b*config.rpm_bin_width:g}, {(b+1)*config.rpm_bin_width:g})" for b in matrix.columns],
                           rotation=45, ha="left", fontsize=8)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels([f"[{b*config.load_bin_width:g}, {(b+1)*config.load_bin_width:g})" for b in matrix.index], fontsize=8)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("RPM — khoảng [cận dưới, cận trên)")
        ax.set_ylabel("Load gốc — thấp ở trên, cao ở dưới")
        ax.set_title(f"{result['dataset']} — {result['metric']} AFR: bảng ước lượng đầy đủ\n"
                     "Không ký hiệu: ô đủ dữ liệu | I: nội suy | E: ngoại suy tuyến tính", pad=18)
        ax.set_xticks(np.arange(-.5, len(matrix.columns)), minor=True)
        ax.set_yticks(np.arange(-.5, len(matrix.index)), minor=True)
        ax.grid(which="minor", color="#94a3b8", linewidth=.4)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row, col in np.ndindex(values.shape):
            red, green, blue, _ = cmap(norm(values[row, col]))
            color = "black" if .2126*red + .7152*green + .0722*blue > .5 else "white"
            ax.text(col, row, f"{values[row,col]:.1f}{marker[origins[row,col]]}",
                    ha="center", va="center", fontsize=7.5, color=color)
        fig.colorbar(image, ax=ax, label="AFR" if result["metric"] != "error" else "Sai lệch AFR")
        fig.savefig(completion["output_dir"] / f"{key}_full.png", dpi=150, bbox_inches="tight")
        figures[key] = fig
    # Một hình riêng để phần lớn ô ngoại suy không bị che bởi màu AFR.
    result = completion["results"]["all_50_trips_target"]
    matrix = result["full"]
    codes = result["audit"].origin.map({"supported": 0, "linear_interpolation": 1, "linear_extrapolation": 2})
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.imshow(codes.to_numpy().reshape(matrix.shape), origin="upper", aspect="auto",
              cmap=ListedColormap(origin_colors), norm=BoundaryNorm([-.5,.5,1.5,2.5],3), interpolation="nearest")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([str(int((b+.5)*config.rpm_bin_width)) for b in matrix.columns], rotation=90)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([str(int((b+.5)*config.load_bin_width)) for b in matrix.index])
    ax.set(xlabel="RPM — tâm vùng", ylabel="Load gốc — tâm vùng", title="Nguồn của từng ô Target AFR")
    ax.legend(handles=[Patch(color=c, label=l) for c,l in zip(origin_colors,
              ["Đủ dữ liệu", "Nội suy tuyến tính", "Ngoại suy tuyến tính"])], loc="upper right")
    fig.savefig(completion["output_dir"] / "target_cell_origin.png", dpi=150, bbox_inches="tight")
    figures["cell_origin"] = fig
    return figures
