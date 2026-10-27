"""Update the existing notebook to annotated RPM/Load AFR tables."""
from pathlib import Path
import json
import shutil
import textwrap

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'main_copy.ipynb'
backup = ROOT / '_notebook_work' / 'main_copy.before_afr_map.ipynb'
if not backup.exists():
    shutil.copy2(path, backup)
nb = json.loads(path.read_text(encoding='utf-8'))
by_id = {c['id']: c for c in nb['cells']}

def replace(name, text):
    c = by_id[name]
    c['source'] = textwrap.dedent(text).strip().splitlines(keepends=True)
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None

first = ''.join(nb['cells'][0]['source'])
if 'STOICH_AFR =' not in first:
    first = first.replace('import matplotlib.pyplot as plt\n', '''import matplotlib.pyplot as plt

# Quy đổi theo giả thiết đã thống nhất: raw = lambda × 1000; xăng chuẩn 14,7.
# Giữ nguyên các cột gốc, chỉ tạo thêm cột AFR suy ra.
STOICH_AFR = 14.7
LAMBDA_RAW_SCALE = 1000.0

def raw_to_afr(values):
    return values / LAMBDA_RAW_SCALE * STOICH_AFR

''')
    replace(nb['cells'][0]['id'], first)

replace('4d427696', '''
plt.figure(figsize=(12, 4))
plt.scatter(df_work["elapsed_s"], raw_to_afr(df_work["LambdaSensor1"]),
            s=2, alpha=0.5, label="AFR cảm biến (quy đổi)")
plt.scatter(df_work["elapsed_s"], raw_to_afr(df_work["TargetAFR"]),
            s=5, color="red", label="AFR mục tiêu (quy đổi)")
plt.axhline(STOICH_AFR, color="black", linestyle="--", linewidth=1,
            label=f"Mốc stoichiometric {STOICH_AFR:g}")
plt.xlabel("Thời gian từ đầu chuyến (giây)")
plt.ylabel("AFR (kg không khí / kg nhiên liệu)")
plt.title("AFR cảm biến và AFR mục tiêu — các giá trị có ghi trong dữ liệu gốc")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
''')

replace('dfb1f016', '''
afr_check = df1[["TargetAFR", "LambdaSensor1", "AFRDifference"]].dropna().copy()
afr_check["difference_calculated_raw"] = (
    afr_check["TargetAFR"] - afr_check["LambdaSensor1"]
)
afr_check["matches_raw"] = (
    afr_check["difference_calculated_raw"] == afr_check["AFRDifference"]
)
afr_check["target_afr"] = raw_to_afr(afr_check["TargetAFR"])
afr_check["measured_afr"] = raw_to_afr(afr_check["LambdaSensor1"])
afr_check["target_minus_sensor_afr"] = (
    afr_check["target_afr"] - afr_check["measured_afr"]
)
afr_check[["target_afr", "measured_afr", "target_minus_sensor_afr", "matches_raw"]].head(10)
''')

replace('fuel-map-intro', r'''
## 9. Bảng Target AFR và AFR cảm biến theo RPM × Load

Hình tham khảo là **bảng AFR mục tiêu (Target AFR)**. Bản đồ chính ở đây dùng RPM
làm cột, Load làm hàng, mỗi ô là trung vị AFR mục tiêu tại vùng đó. Bảng cảm biến
cho biết AFR đo được; bảng sai lệch giúp đối chiếu. `BaseFuel` là một kênh khác,
không còn là đại lượng trong các ô bản đồ AFR.

Theo giả thiết đã thống nhất cho bộ log này:

$$\lambda=\frac{raw}{1000},\qquad AFR=\frac{raw}{1000}\times14.7.$$

Ví dụ 1000 → 1,000 lambda → 14,7 AFR; 920 → 0,920 lambda → 13,524 AFR.
Hệ số được khai báo một lần ở cell đầu. Các cột gốc được bảo toàn; bổ sung cột
`target_lambda`, `measured_lambda`, `target_afr`, `measured_afr`.
Mốc 14,7 chỉ là mốc tham chiếu; không ép toàn bộ `TargetAFR` về 14,7.

Quy trình: kiểm tra → gộp thời gian trùng → giữ giá trị trước có giới hạn
→ lưới đều → quy đổi AFR → chia vùng → thống kê và vẽ bảng có số trong ô.
Ô trống có thể là kênh chưa cập nhật: đây vẫn là giả thiết tái dựng chưa xác nhận
cho định dạng xuất log. Không nội suy các vùng còn thiếu, không sao chép số từ hình mẫu.

Nguồn về quan hệ lambda–AFR: [Haltech CAN Units](https://support.haltech.com/portal/en/kb/articles/haltech-can-protocol-specification).
Đây là bảng mô tả một chuyến dưới các giả thiết trên, chưa phải bảng hiệu chỉnh đã kiểm chứng.
''')

replace('fuel-map-settings', '''
import numpy as np
from IPython.display import display

# Các lựa chọn thực hành, có thể chỉnh rồi chạy lại từ cell này trở xuống.
SAMPLE_PERIOD_S = 0.1
SEGMENT_GAP_S = 2.0
MAX_AGE_S = {
    "RPM": 0.25, "Load": 0.25,
    "LambdaSensor1": 0.25, "TargetAFR": 2.0,
    "CoolantTemp": 2.0, "ThrottlePosition": 0.25,
}
RPM_BIN_WIDTH = 500
LOAD_BIN_WIDTH = 50  # Giữ số đo Load gốc.
MIN_GRID_SAMPLES = 30
MIN_CHANNEL_UPDATES = 5
AFR_COLOR_MIN, AFR_COLOR_MAX = 10.0, 20.0
CORE = ["RPM", "Load"]  # Không buộc có BaseFuel để lập bản đồ AFR.
MAP_CHANNELS = list(MAX_AGE_S)
assert STOICH_AFR > 0 and LAMBDA_RAW_SCALE > 0
assert SAMPLE_PERIOD_S > 0 and SEGMENT_GAP_S > 0
assert RPM_BIN_WIDTH > 0 and LOAD_BIN_WIDTH > 0
assert MIN_GRID_SAMPLES >= 1 and MIN_CHANNEL_UPDATES >= 1
assert AFR_COLOR_MIN < AFR_COLOR_MAX
assert all(value >= 0 for value in MAX_AGE_S.values())
np.testing.assert_allclose(raw_to_afr(1000.0), 14.7)
np.testing.assert_allclose(raw_to_afr(920.0), 13.524)

display(pd.Series(MAX_AGE_S, name="max_age_s").to_frame())
print(f"AFR = raw / {LAMBDA_RAW_SCALE:g} × {STOICH_AFR:g}")
''')

replace('fuel-map-valid-notes', r'''
### 9.4. Quy đổi AFR và kiểm chứng dữ liệu đồng bộ

Chỉ giữ vị trí vận hành có đủ RPM và Load, RPM > 0 và Load ≥ 0. Đối với từng bảng,
chỉ dùng kênh AFR dương và còn trong giới hạn độ cũ; không buộc kênh kia phải có
dữ liệu. Vì vậy bảng mục tiêu và bảng cảm biến có độ phủ khác nhau. Không loại số
đo chỉ vì khác 14,7; không gán giá trị thiếu thành 14,7.

Quy ước sai lệch mới: **AFR cảm biến − AFR mục tiêu**. Giá trị dương là cảm biến
cao hơn mục tiêu. Cột `AFRDifference` gốc có chiều ngược lại (mục tiêu − cảm biến),
nên khi đổi đơn vị để đối chiếu phải đổi cả dấu.
''')

# Update the preview left over from the earlier BaseFuel workflow.
replace('fuel-map-align', ''.join(by_id['fuel-map-align']['source']).replace(
    '["elapsed_s", "RPM", "RPM__age_s", "Load", "BaseFuel"]',
    '["elapsed_s", "RPM", "RPM__age_s", "Load", "TargetAFR", "LambdaSensor1"]'
))

replace('fuel-map-valid', '''
for channel, max_age in MAX_AGE_S.items():
    valid = df_aligned[channel].notna()
    ages = df_aligned.loc[valid, channel + "__age_s"]
    assert ages.ge(-1e-9).all() and ages.le(max_age + 1e-9).all()
    source_time = df_aligned.loc[valid, channel + "__source_elapsed_s"]
    lookup = pd.Series(events[channel].to_numpy(), index=events.index.total_seconds())
    expected = lookup.reindex(source_time.to_numpy()).to_numpy()
    np.testing.assert_allclose(df_aligned.loc[valid, channel].to_numpy(),
                               expected, rtol=0, atol=0)
    source_segments = pd.Series(
        events["segment_id"].to_numpy(), index=events.index.total_seconds()
    ).reindex(source_time.to_numpy()).to_numpy()
    assert (source_segments == df_aligned.loc[valid, "segment_id"].to_numpy()).all()

df_aligned["target_lambda"] = df_aligned["TargetAFR"] / LAMBDA_RAW_SCALE
df_aligned["measured_lambda"] = df_aligned["LambdaSensor1"] / LAMBDA_RAW_SCALE
df_aligned["target_afr"] = raw_to_afr(df_aligned["TargetAFR"])
df_aligned["measured_afr"] = raw_to_afr(df_aligned["LambdaSensor1"])
df_aligned["afr_error"] = df_aligned["measured_afr"] - df_aligned["target_afr"]
for raw_channel, afr_channel in [("TargetAFR", "target_afr"), ("LambdaSensor1", "measured_afr")]:
    assert df_aligned[raw_channel].isna().equals(df_aligned[afr_channel].isna())
    np.testing.assert_allclose(
        df_aligned[afr_channel].dropna(),
        df_aligned.loc[df_aligned[afr_channel].notna(), raw_channel] / LAMBDA_RAW_SCALE * STOICH_AFR,
    )

complete = df_aligned[CORE].notna().all(axis=1)
basic_range = df_aligned["RPM"].gt(0) & df_aligned["Load"].ge(0)
df_aligned["usable_position"] = complete & basic_range
df_aligned["usable_target_afr"] = (
    df_aligned["usable_position"] & df_aligned["target_afr"].gt(0)
)
df_aligned["usable_measured_afr"] = (
    df_aligned["usable_position"] & df_aligned["measured_afr"].gt(0)
)
map_points = df_aligned.loc[df_aligned["usable_position"]].copy()
assert not map_points.empty, "Không có vị trí RPM–Load hợp lệ."
for channel, raw_channel, flag in [
    ("target_afr", "TargetAFR", "usable_target_afr"),
    ("measured_afr", "LambdaSensor1", "usable_measured_afr"),
]:
    map_points[channel] = map_points[channel].where(map_points[flag])
    map_points[channel + "__source_s"] = map_points[raw_channel + "__source_elapsed_s"].where(
        map_points[channel].notna()
    )
paired = map_points[["target_afr", "measured_afr"]].notna().all(axis=1)
map_points["afr_error"] = (map_points["measured_afr"] - map_points["target_afr"]).where(paired)
map_points["paired_target_source_s"] = map_points["target_afr__source_s"].where(paired)
map_points["paired_measured_source_s"] = map_points["measured_afr__source_s"].where(paired)

selection_report = pd.Series({
    "grid_rows": len(df_aligned),
    "missing_position_rows": (~complete).sum(),
    "complete_position_outside_basic_range": (complete & ~basic_range).sum(),
    "valid_position_rows": len(map_points),
    "valid_target_rows": map_points["target_afr"].notna().sum(),
    "valid_measured_rows": map_points["measured_afr"].notna().sum(),
    "valid_paired_rows": paired.sum(),
}, name="value")
display(selection_report.to_frame())
display(df_aligned[["TargetAFR", "target_afr", "LambdaSensor1", "measured_afr", "afr_error"]].head(12).round(4))

fig_afr_time, ax = plt.subplots(figsize=(12, 4))
ax.scatter(df_aligned["elapsed_s"], df_aligned["measured_afr"], s=2, alpha=0.45,
           label="AFR cảm biến, đã đồng bộ")
ax.scatter(df_aligned["elapsed_s"], df_aligned["target_afr"], s=4, color="red",
           label="AFR mục tiêu, đã đồng bộ")
ax.axhline(STOICH_AFR, color="black", linestyle="--", linewidth=1,
           label=f"Mốc {STOICH_AFR:g}")
ax.set(xlabel="Giây từ đầu chuyến", ylabel="AFR", title="AFR quy đổi theo thời gian")
ax.grid(alpha=0.25)
ax.legend()
fig_afr_time.tight_layout()
plt.show()
''')

replace('fuel-map-bins-notes', r'''
### 9.5. Thống kê AFR theo vùng RPM × Load

Khoảng RPM rộng 500, khoảng Load rộng 50, đóng trái và mở phải `[a, b)`.
Mỗi ô của bảng mục tiêu là:

$$T_{ij}=\operatorname{median}\{AFR_{target}(t):t\in B_{ij}\}.$$

Tương tự, $M_{ij}$ là trung vị AFR cảm biến. Sai lệch được tính trên **các cặp tại
cùng mốc** rồi lấy trung vị, không lấy hiệu của hai trung vị có tập mẫu khác nhau.
Mỗi bảng có số mốc và số thời điểm nguồn riêng. Chỉ hiển thị ô có ≥30 mốc lưới
và ≥5 thời điểm ghi nguồn của kênh đó. Đây là bộ lọc thưa để thực hành, không phải
kiểm định độ tin cậy. Các mốc lưới và thời điểm nguồn không bảo đảm độc lập.
''')

replace('fuel-map-bins', '''
rpm_edges = np.arange(
    np.floor(map_points["RPM"].min() / RPM_BIN_WIDTH) * RPM_BIN_WIDTH,
    (np.floor(map_points["RPM"].max() / RPM_BIN_WIDTH) + 2) * RPM_BIN_WIDTH,
    RPM_BIN_WIDTH,
)
load_edges = np.arange(
    np.floor(map_points["Load"].min() / LOAD_BIN_WIDTH) * LOAD_BIN_WIDTH,
    (np.floor(map_points["Load"].max() / LOAD_BIN_WIDTH) + 2) * LOAD_BIN_WIDTH,
    LOAD_BIN_WIDTH,
)
map_points["rpm_bin"] = pd.cut(map_points["RPM"], rpm_edges, right=False)
map_points["load_bin"] = pd.cut(map_points["Load"], load_edges, right=False)
assert map_points[["rpm_bin", "load_bin"]].notna().all().all()

cell_stats = map_points.groupby(["load_bin", "rpm_bin"], observed=True).agg(
    n_grid=("RPM", "size"),
    n_target=("target_afr", "count"),
    n_target_updates=("target_afr__source_s", "nunique"),
    target_afr_median=("target_afr", "median"),
    target_afr_q25=("target_afr", lambda x: x.quantile(0.25)),
    target_afr_q75=("target_afr", lambda x: x.quantile(0.75)),
    n_measured=("measured_afr", "count"),
    n_measured_updates=("measured_afr__source_s", "nunique"),
    measured_afr_median=("measured_afr", "median"),
    measured_afr_q25=("measured_afr", lambda x: x.quantile(0.25)),
    measured_afr_q75=("measured_afr", lambda x: x.quantile(0.75)),
    n_paired=("afr_error", "count"),
    n_paired_target_updates=("paired_target_source_s", "nunique"),
    n_paired_measured_updates=("paired_measured_source_s", "nunique"),
    afr_error_median=("afr_error", "median"),
)
for prefix in ["target", "measured"]:
    cell_stats[prefix + "_represented_time_s"] = cell_stats["n_" + prefix] * SAMPLE_PERIOD_S
    cell_stats[prefix + "_afr_iqr"] = (
        cell_stats[prefix + "_afr_q75"] - cell_stats[prefix + "_afr_q25"]
    )
    cell_stats[prefix + "_supported"] = (
        cell_stats["n_" + prefix].ge(MIN_GRID_SAMPLES)
        & cell_stats["n_" + prefix + "_updates"].ge(MIN_CHANNEL_UPDATES)
    )
cell_stats["paired_supported"] = (
    cell_stats["n_paired"].ge(MIN_GRID_SAMPLES)
    & cell_stats["n_paired_target_updates"].ge(MIN_CHANNEL_UPDATES)
    & cell_stats["n_paired_measured_updates"].ge(MIN_CHANNEL_UPDATES)
)
rpm_intervals = pd.IntervalIndex.from_breaks(rpm_edges, closed="left")
load_intervals = pd.IntervalIndex.from_breaks(load_edges, closed="left")

def map_matrix(column):
    return cell_stats[column].unstack("rpm_bin").reindex(
        index=load_intervals, columns=rpm_intervals
    )

coverage_map = map_matrix("n_grid").fillna(0).astype(int)
target_count_map = map_matrix("n_target").fillna(0).astype(int)
measured_count_map = map_matrix("n_measured").fillna(0).astype(int)
paired_count_map = map_matrix("n_paired").fillna(0).astype(int)
target_support_map = map_matrix("target_supported").eq(True)
measured_support_map = map_matrix("measured_supported").eq(True)
paired_support_map = map_matrix("paired_supported").eq(True)
target_afr_map_all = map_matrix("target_afr_median")
measured_afr_map_all = map_matrix("measured_afr_median")
target_afr_map = target_afr_map_all.where(target_support_map)
measured_afr_map = measured_afr_map_all.where(measured_support_map)
afr_error_map = map_matrix("afr_error_median").where(paired_support_map)

assert coverage_map.to_numpy().sum() == len(map_points)
assert target_count_map.to_numpy().sum() == map_points["target_afr"].notna().sum()
assert measured_count_map.to_numpy().sum() == map_points["measured_afr"].notna().sum()
assert paired_count_map.to_numpy().sum() == map_points["afr_error"].notna().sum()
assert target_afr_map.notna().equals(target_support_map)
assert measured_afr_map.notna().equals(measured_support_map)
assert afr_error_map.notna().equals(paired_support_map)

map_summary = pd.DataFrame({
    "AFR_table": ["Target AFR", "Measured AFR", "Measured - Target"],
    "valid_grid_rows": [cell_stats.n_target.sum(), cell_stats.n_measured.sum(), cell_stats.n_paired.sum()],
    "observed_cells": [(target_count_map > 0).to_numpy().sum(),
                       (measured_count_map > 0).to_numpy().sum(),
                       (paired_count_map > 0).to_numpy().sum()],
    "displayed_cells": [target_support_map.to_numpy().sum(),
                        measured_support_map.to_numpy().sum(),
                        paired_support_map.to_numpy().sum()],
    "total_cells": coverage_map.size,
})
display(map_summary, cell_stats.sort_values("n_target", ascending=False).head(10).round(3))
''')

replace('fuel-map-plot-notes', '''
### 9.6. Bảng AFR có số trong ô, theo hình tham khảo

Bảng chính là **Target AFR**, bảng thứ hai là **AFR cảm biến**. Cột là RPM tăng
từ trái sang phải; hàng là Load tăng từ trên xuống dưới, giống bố cục hình mẫu.
Nhãn là khoảng giá trị `[a,b)` chứ không phải điểm đặt chính xác của ECU.
Các ô hiện một chữ số thập phân; bảng CSV giữ độ chính xác đầy đủ.
Hai bảng dùng chung thang màu 10–20 AFR; số được ghi đúng ngay cả khi ngoài thang màu.
Ô xám có dấu “—” là chưa đủ dữ liệu theo quy tắc lọc thưa, không phải AFR bằng 0.
''')

replace('fuel-map-plots', '''
from matplotlib.colors import Normalize, TwoSlopeNorm

def draw_afr_table(ax, matrix, title, *, cmap_name="turbo", norm=None):
    values = matrix.to_numpy(dtype=float)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#e5e7eb")
    if norm is None:
        norm = Normalize(vmin=AFR_COLOR_MIN, vmax=AFR_COLOR_MAX, clip=True)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm,
                      aspect="auto", origin="upper", interpolation="nearest")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([f"[{v.left:g}, {v.right:g})" for v in matrix.columns],
                       rotation=45, ha="left", fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([f"[{v.left:g}, {v.right:g})" for v in matrix.index], fontsize=9)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("RPM — khoảng [cận dưới, cận trên)", labelpad=10)
    ax.set_ylabel("Load gốc — thấp ở trên, cao ở dưới")
    ax.set_title(title, fontsize=12, pad=18)
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="#64748b", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                red, green, blue, _ = cmap(norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                color = "black" if luminance > 0.5 else "white"
                ax.text(col, row, f"{value:.1f}", ha="center", va="center",
                        fontsize=9, color=color)
            else:
                ax.text(col, row, "—", ha="center", va="center", fontsize=8, color="#64748b")
    return image

fig_target, ax_target = plt.subplots(figsize=(14, 10), constrained_layout=True)
target_image = draw_afr_table(
    ax_target, target_afr_map, f"Target AFR map — {file.stem}"
)
fig_target.colorbar(target_image, ax=ax_target, label="AFR mục tiêu")
plt.show()

fig_measured, ax_measured = plt.subplots(figsize=(14, 10), constrained_layout=True)
measured_image = draw_afr_table(
    ax_measured, measured_afr_map, f"Measured AFR map — {file.stem}"
)
fig_measured.colorbar(measured_image, ax=ax_measured, label="AFR cảm biến")
plt.show()

finite_error = np.abs(afr_error_map.to_numpy(dtype=float))
error_limit = max(0.5, float(np.nanmax(finite_error))) if np.isfinite(finite_error).any() else 0.5
fig_error, ax_error = plt.subplots(figsize=(14, 10), constrained_layout=True)
error_image = draw_afr_table(
    ax_error, afr_error_map, "AFR cảm biến − AFR mục tiêu (trung vị các cặp cùng mốc)",
    cmap_name="RdBu_r", norm=TwoSlopeNorm(vmin=-error_limit, vcenter=0, vmax=error_limit),
)
fig_error.colorbar(error_image, ax=ax_error, label="Sai lệch AFR")
plt.show()
''')

replace('fuel-map-baseline-notes', '''
### 9.7. Đối chiếu với các dòng có dữ liệu gốc

Tính riêng cho từng kênh trên các dòng có đủ RPM, Load và kênh AFR tương ứng,
không điền khuyết, dùng cùng vùng. So sánh ảnh hưởng của tái dựng và lấy mẫu đều.
Hai cách có cách chọn và trọng số khác nhau; chênh lệch không tự chứng minh một
cách đúng hơn. Cột chênh lệch có đơn vị AFR sau quy đổi.
''')

replace('fuel-map-baseline', '''
raw_baseline_points = numeric.copy()
raw_baseline_points["target_afr"] = raw_to_afr(raw_baseline_points["TargetAFR"])
raw_baseline_points["measured_afr"] = raw_to_afr(raw_baseline_points["LambdaSensor1"])
comparison = cell_stats.copy()
for prefix, afr_channel in [("target", "target_afr"), ("measured", "measured_afr")]:
    raw_subset = raw_baseline_points[["RPM", "Load", afr_channel]].dropna().copy()
    raw_subset = raw_subset.loc[
        raw_subset["RPM"].gt(0) & raw_subset["Load"].ge(0) & raw_subset[afr_channel].gt(0)
    ]
    raw_subset["rpm_bin"] = pd.cut(raw_subset["RPM"], rpm_edges, right=False)
    raw_subset["load_bin"] = pd.cut(raw_subset["Load"], load_edges, right=False)
    baseline = raw_subset.groupby(["load_bin", "rpm_bin"], observed=True)[afr_channel].agg(
        ["count", "median"]
    ).rename(columns={"count": prefix + "_raw_rows", "median": prefix + "_raw_median"})
    comparison = comparison.join(baseline, how="left")
    comparison[prefix + "_aligned_minus_raw"] = (
        comparison[prefix + "_afr_median"] - comparison[prefix + "_raw_median"]
    )
display(comparison[["n_target", "target_raw_rows", "target_afr_median", "target_raw_median",
                    "target_aligned_minus_raw", "n_measured", "measured_raw_rows",
                    "measured_afr_median", "measured_raw_median", "measured_aligned_minus_raw"]]
        .sort_values("n_target", ascending=False).head(12).round(3))
''')

replace('fuel-map-afr-notes', '''
### 9.8. Độ phủ riêng và kiểm tra công thức sai lệch

Ba bảng đếm có mẫu số riêng: mục tiêu, cảm biến và cặp đủ cả hai kênh.
Số mốc sau giữ giá trị không phải số phép đo độc lập. Bảng Target AFR tái dựng
những mục tiêu đã được ghi, chưa tự đề xuất mục tiêu mới hay hiệu chỉnh BaseFuel.
''')

replace('fuel-map-afr', '''
from matplotlib.colors import LogNorm

fig_coverage, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
for ax, matrix, title in zip(axes,
    [target_count_map, measured_count_map, paired_count_map],
    ["Độ phủ AFR mục tiêu", "Độ phủ AFR cảm biến", "Độ phủ các cặp cùng mốc"]):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e5e7eb")
    image = ax.imshow(np.ma.masked_where(matrix.to_numpy() == 0, matrix.to_numpy()),
                      origin="upper", aspect="auto", interpolation="nearest", cmap=cmap,
                      norm=LogNorm(vmin=1, vmax=max(2, int(matrix.to_numpy().max()))))
    ax.set_xticks(np.arange(len(rpm_intervals)))
    ax.set_xticklabels([f"{v.mid:g}" for v in rpm_intervals], rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(load_intervals)))
    ax.set_yticklabels([f"{v.mid:g}" for v in load_intervals], fontsize=8)
    ax.set(xlabel="RPM — tâm vùng", ylabel="Load gốc — tâm vùng", title=title)
    fig_coverage.colorbar(image, ax=ax, label=f"Số mốc {SAMPLE_PERIOD_S:g} s")
plt.show()

# Kiểm tra cả đơn vị lẫn chiều dấu trên những dòng gốc đủ ba kênh.
check = df1[["TargetAFR", "LambdaSensor1", "AFRDifference"]].dropna()
converted_error = raw_to_afr(check["LambdaSensor1"]) - raw_to_afr(check["TargetAFR"])
expected_error = -raw_to_afr(check["AFRDifference"])
error_formula_report = pd.Series({
    "checked_raw_rows": len(check),
    "matching_rows": np.isclose(converted_error, expected_error).sum(),
    "max_abs_disagreement": (converted_error - expected_error).abs().max(),
}, name="value")
display(error_formula_report.to_frame())
assert len(check) > 0
np.testing.assert_allclose(converted_error, expected_error, atol=1e-12, rtol=1e-12)
''')

replace('fuel-map-export-notes', '''
### 9.9. Lưu bảng AFR và hình minh họa

Kết quả mới lưu trong `outputs/<tên chuyến>/afr_maps/` để phân biệt với kết quả
BaseFuel trước đây. Bảng `target_afr_map.csv` là bảng mục tiêu theo hình tham khảo;
`measured_afr_map.csv` là bảng cảm biến; `afr_error_map.csv` là sai lệch cảm biến
trừ mục tiêu. Các ô thiếu vẫn trống trong CSV. Cột gốc, cột đã quy đổi, tuổi giá trị
và cờ xử lý đều có trong `aligned_data.csv`.

Đọc kết quả: xem độ phủ trước, sau đó xem AFR mục tiêu, AFR cảm biến và sai lệch.
Chưa bù độ trễ cảm biến hoặc chọn riêng trạng thái động cơ nóng/vận hành ổn định.
Không thay ô thiếu bằng 14,7 hoặc tự kéo giá trị sang các vùng chưa quan sát.
''')

replace('fuel-map-export', '''
import json

output_dir = Path("outputs") / file.stem / "afr_maps"
output_dir.mkdir(parents=True, exist_ok=True)
df_aligned.to_csv(output_dir / "aligned_data.csv", index_label="elapsed")
input_quality.to_csv(output_dir / "input_quality.csv", index_label="channel")
alignment_report.to_csv(output_dir / "alignment_report.csv", index_label="channel")
comparison.to_csv(output_dir / "afr_cell_statistics.csv")
map_summary.to_csv(output_dir / "map_summary.csv", index=False)
for name, matrix in {
    "target_afr_map": target_afr_map,
    "measured_afr_map": measured_afr_map,
    "afr_error_map": afr_error_map,
    "target_afr_map_all_observed": target_afr_map_all,
    "measured_afr_map_all_observed": measured_afr_map_all,
    "target_coverage": target_count_map,
    "measured_coverage": measured_count_map,
    "paired_coverage": paired_count_map,
}.items():
    matrix.to_csv(output_dir / f"{name}.csv", index_label="Load_bin")

for name, figure in {
    "target_afr_map": fig_target,
    "measured_afr_map": fig_measured,
    "afr_error_map": fig_error,
    "afr_coverage": fig_coverage,
    "afr_timeseries": fig_afr_time,
}.items():
    figure.savefig(output_dir / f"{name}.png", dpi=160, bbox_inches="tight")

settings = {
    "source_file": str(file), "source_rows": len(df_raw),
    "stoich_afr": STOICH_AFR, "lambda_raw_scale": LAMBDA_RAW_SCALE,
    "conversion_formula": "AFR = raw / lambda_raw_scale * stoich_afr",
    "conversion_status": "user-approved assumption for this ECU Manager CSV",
    "sample_period_s": SAMPLE_PERIOD_S, "segment_gap_s": SEGMENT_GAP_S,
    "max_age_s": MAX_AGE_S, "rpm_bin_width": RPM_BIN_WIDTH,
    "load_bin_width": LOAD_BIN_WIDTH, "min_grid_samples": MIN_GRID_SAMPLES,
    "min_channel_updates": MIN_CHANNEL_UPDATES,
    "afr_color_range": [AFR_COLOR_MIN, AFR_COLOR_MAX],
    "duplicate_time_rule": "last non-null per channel in original row order",
    "missing_value_assumption": "carry last observation, bounded age; unverified export semantics",
    "load_units": "unchanged raw values",
    "bin_intervals": "left closed, right open; Load low at top",
    "error_sign": "measured AFR minus target AFR",
    "purpose": "descriptive target/measured AFR tables from one trip; no BaseFuel correction",
}
(output_dir / "settings.json").write_text(
    json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
)
pd.testing.assert_frame_equal(df1, df_raw[df1.columns])
assert target_afr_map.notna().equals(target_support_map)
assert measured_afr_map.notna().equals(measured_support_map)
for name, matrix in [("target_afr_map", target_afr_map), ("measured_afr_map", measured_afr_map)]:
    saved = pd.read_csv(output_dir / f"{name}.csv", index_col=0)
    np.testing.assert_allclose(saved.to_numpy(), matrix.to_numpy(), equal_nan=True)
display(map_summary)
print("Đã bảo toàn dữ liệu gốc, kiểm tra công thức AFR, dấu sai lệch và bảng CSV đã lưu.")
print("Kết quả:", output_dir.resolve())
''')

# No old chart outputs remain visible under the revised formulas until re-execution.
for c in nb['cells']:
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(f'Updated AFR notebook: {path}')
