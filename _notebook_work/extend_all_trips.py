from pathlib import Path
import json
import shutil
import textwrap

root = Path(__file__).resolve().parents[1]
path = root / 'main_copy.ipynb'
backup = root / '_notebook_work/main_copy.before_all_trips.ipynb'
if not backup.exists():
    shutil.copy2(path, backup)
nb = json.loads(path.read_text(encoding='utf-8'))

def cell(kind, source, name):
    result = {'cell_type': kind, 'id': name, 'metadata': {},
              'source': textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result

def md(source, name):
    return cell('markdown', source, name)

def code(source, name):
    return cell('code', source, name)

new = [
md(r'''
## 10. Tổng hợp cả 50 chuyến: bảng AFR và độ phủ giữa các chuyến

Phần 9 giữ lại ví dụ một chuyến. Phần 10 áp dụng cùng giả thiết quy đổi và điền
khuyết cho toàn bộ `car_data`, đọc metadata riêng của mỗi CSV (có 3 cấu trúc cột).
Các hàm đọc, đồng bộ, thống kê, kiểm chứng và vẽ nằm trong `afr_pipeline.py` để có
thể xem lại và tái sử dụng. Mỗi chuyến xử lý độc lập, không điền xuyên chuyến hay
khoảng gián đoạn dài. Giá trị cảm biến bằng 0 được giữ trong bảng kiểm tra nhưng
không dùng để tính AFR cảm biến hoặc sai lệch.

**Cách gộp:** trong mỗi ô, lấy trung vị AFR của từng chuyến đủ dữ liệu, sau đó lấy
trung vị giữa các chuyến. Mỗi chuyến có một trọng số như nhau trong ô:

$$m_{kij}=\operatorname{median}(AFR\text{ trong chuyến }k\text{, vùng }ij),$$
$$M_{ij}=\operatorname{median}_{k\in K_{ij}}(m_{kij}).$$

Một chuyến đóng góp khi vùng có ít nhất 30 mốc 0,1 giây và 5 lần ghi nguồn của
kênh đó. Ô tổng hợp hiển thị khi có ít nhất 3 chuyến đủ điều kiện. Số thời điểm
nguồn được đếm riêng trong từng chuyến, không gộp nhầm cùng giây ở hai file.
Đây là quy tắc thực hành có thể chỉnh, chưa phải bằng chứng hiệu chỉnh động cơ tối ưu.

Mục tiêu AFR có thay đổi theo thời gian. Vì vậy xuất cả bảng tổng hợp, số chuyến,
IQR/P90–P10 giữa chuyến và bảng theo 3 giai đoạn để khảo sát. Các nhóm thời gian
không khẳng định là 3 phiên bản hiệu chỉnh ECU; phân bố RPM/Load trong vùng cũng
có thể ảnh hưởng kết quả. Không nội suy các ô thiếu và không điền chúng bằng 14,7.
''', 'multi-afr-intro'),
code('''
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from afr_pipeline import AFRConfig, run_all_trips, plot_tables, matrix_for

multi_config = AFRConfig(
    stoich_afr=14.7,
    raw_lambda_scale=1000.0,
    sample_period_s=0.1,
    segment_gap_s=2.0,
    max_age_s={"RPM": 0.25, "Load": 0.25, "TargetAFR": 2.0, "LambdaSensor1": 0.25},
    rpm_bin_width=500,
    load_bin_width=50,
    min_trip_samples=30,
    min_trip_updates=5,
    min_trips=3,
    target_spread_flag_afr=0.3,
    holdout_date_fraction=0.2,
)
multi_files = sorted(Path("car_data").glob("*.csv"))
assert len(multi_files) == 50, f"Đang có {len(multi_files)} file, cần kiểm tra phạm vi."
multi_output = Path("outputs/all_trips_afr")
multi_result = run_all_trips("car_data", multi_output, multi_config)
assert len(multi_result["trips"]) == len(multi_files) == 50
assert multi_result["trips"].trip_id.is_unique
print("Đã xử lý đủ 50 file; kết quả:", multi_output.resolve())
''', 'multi-afr-run'),
md('''
### 10.1. Báo cáo chất lượng của toàn bộ nguồn

Xem số dòng, cấu trúc, đoạn ghi gián đoạn, giá trị 0 của cảm biến và tỷ lệ thiếu
theo từng chuyến. `AFRDifference` là kênh kiểm tra phụ; các dòng không khớp được
báo cáo, không loại cả chuyến. Việc gộp mốc trùng vẫn giữ quy ước giá trị không
khuyết cuối của từng kênh, còn file nguồn được giữ nguyên và có SHA-256 để truy vết.
''', 'multi-afr-quality-notes'),
code('''
multi_trips = multi_result["trips"]
multi_cells = multi_result["trip_cells"]
multi_summary = multi_result["summary"]
multi_overview = pd.Series({
    "files": len(multi_trips),
    "raw_rows": multi_trips.raw_rows.sum(),
    "grid_rows": multi_trips.grid_rows.sum(),
    "recording_dates": multi_trips.log_date.nunique(),
    "schema_count": multi_trips.schema.nunique(),
    "extra_duplicate_time_rows": multi_trips.extra_duplicate_time_rows.sum(),
    "recording_breaks": multi_trips.recording_breaks.sum(),
    "sensor_zero_raw_rows": multi_trips.zero_sensor_raw_rows.sum(),
    "AFRDifference_checked": multi_trips.difference_checked_rows.sum(),
    "AFRDifference_mismatches": multi_trips.difference_mismatch_rows.sum(),
}, name="value")
display(multi_overview.to_frame())
display(multi_trips[["trip_id", "raw_rows", "column_count", "grid_rows", "largest_gap_s",
                     "valid_target_rows", "valid_measured_rows", "difference_mismatch_rows"]].head(12))
display(multi_trips.loc[multi_trips.recording_breaks.gt(0),
                       ["trip_id", "largest_gap_s", "recording_breaks", "segments"]])
display(multi_result["channel_quality"].groupby("channel")[[
    "raw_missing_pct", "grid_missing_pct", "grid_carried_pct"
]].mean().rename(columns=lambda x: "mean_trip_" + x).round(2))
''', 'multi-afr-quality'),
md('''
### 10.2. Bảng AFR tổng hợp có kiểm tra số chuyến hỗ trợ

Bảng mục tiêu và cảm biến có bộ lọc riêng. Sai lệch là `measured − target` trên
cặp cùng mốc, tính trong từng chuyến trước khi tổng hợp. Không lấy hiệu hai bảng
trung vị có các tập mẫu khác nhau.

Dấu `*` trong bảng Target AFR nghĩa là P90–P10 của trung vị giữa các chuyến lớn
hơn 0,3 AFR: các chuyến có mục tiêu khác nhau đáng xem xét. Đây là cờ khảo sát,
không tự loại các chuyến hoặc kết luận cảm biến lỗi. Các ô thiếu hoặc chưa đủ 3
chuyến vẫn để trống. Bảng `all_observed_trip_median` lưu cả vùng chưa đạt hỗ trợ
để xem thêm, tách khỏi bảng chính.
''', 'multi-afr-maps-notes'),
code('''
multi_map_overview = multi_summary.groupby("metric").agg(
    cells_observed=("n_trips_observed", "size"),
    cells_supported=("supported", "sum"),
    target_variable_cells=("target_variable", "sum"),
    max_supporting_trips=("n_trips", "max"),
)
multi_map_overview["grid_cells"] = len(multi_result["rpm_bins"]) * len(multi_result["load_bins"])
multi_map_overview["supported_grid_pct"] = (
    multi_map_overview.cells_supported / multi_map_overview.grid_cells * 100
)
display(multi_map_overview.round(2))
display(multi_summary.loc[multi_summary.metric.eq("target") & multi_summary.target_variable,
    ["load_bin", "rpm_bin", "value", "n_trips", "n_dates", "trip_iqr", "trip_p90_p10"]]
    .sort_values("trip_p90_p10", ascending=False).head(12).round(3))
multi_figures = plot_tables(multi_result)
plt.show()
''', 'multi-afr-maps'),
md('''
### 10.3. Kiểm tra mục tiêu thay đổi theo thời gian

Các file ghi nhận mức TargetAFR cao nhất lần lượt là 1000, 1068 và 1088 trong
ba giai đoạn. Sau quy đổi, các giá trị đó là 14,7; 15,6996 và 15,9936 AFR.
Đây là các cực đại quan sát, không phải mục tiêu ở mọi ô. So sánh những vùng
được hỗ trợ ở cả ba giai đoạn để giảm việc so sánh các vùng không tương ứng.
Không trộn các giá trị đó thành một bảng được tuyên bố là mục tiêu tốt nhất cho xe.
''', 'multi-afr-period-notes'),
code('''
multi_periods = multi_result["period_comparison"]
multi_period_report = multi_periods.groupby("period").agg(
    observed_cells=("value", "size"), supported_cells=("supported", "sum"),
)
display(multi_period_report)
period_values = multi_periods.loc[multi_periods.supported].pivot(
    index=["load_bin", "rpm_bin"], columns="period", values="value"
)
period_common = period_values.dropna().copy()
period_common["period_max_minus_min_afr"] = period_common.max(axis=1) - period_common.min(axis=1)
period_common.to_csv(multi_output / "common_cells_period_comparison.csv")
display(period_common.sort_values("period_max_minus_min_afr", ascending=False).head(15).round(3))
print("Số vùng được hỗ trợ ở cả ba giai đoạn:", len(period_common))
print("Hình và CSV riêng từng giai đoạn đã được lưu trong", multi_output)
''', 'multi-afr-period'),
md('''
### 10.4. Kiểm chứng bằng các ngày chưa dùng để xây bảng

Giữ riêng khoảng 20% ngày cuối, bảo đảm hai chiều đi cùng ngày không bị chia sang
cả tập xây bảng và kiểm chứng. Chỉ dùng các ngày trước để lập bảng dự đoán.
Đối chiếu với trung vị chuyến×vùng của ngày giữ lại; tính MAE và bias trong từng
chuyến rồi trung bình giữa chuyến. Coverage là tỷ lệ vùng kiểm chứng đủ điều kiện
có giá trị dự đoán từ bảng trước đó. Vùng không dự đoán được vẫn được tính trong
mẫu số coverage, không biến thành sai số bằng 0.

Đây là kiểm chứng khả năng tái hiện số đo lịch sử, không chứng minh hiệu chỉnh
động cơ đúng. Mục tiêu có thể thay đổi theo thời gian. Sau kiểm chứng, bảng cuối
được xây lại từ cả 50 chuyến; thông tin ngày giữ lại vẫn được lưu trong settings.
''', 'multi-afr-validation-notes'),
code('''
print("Ngày giữ lại:", ", ".join(multi_result["heldout_dates"]))
display(multi_result["validation"].round(4))
assert multi_result["validation"].coverage_pct.between(0, 100).all()
assert multi_result["validation"].predicted_trip_cells.le(
    multi_result["validation"].eligible_test_trip_cells
).all()
''', 'multi-afr-validation'),
md('''
### 10.5. Kiểm tra kết quả đã lưu và các bảng cần đọc

Các file chính trong `outputs/all_trips_afr/`:

- `target_afr_map.csv` và `.png`: mục tiêu tổng hợp, cân bằng giữa chuyến.
- `measured_afr_map.csv` và `.png`: AFR cảm biến tổng hợp.
- `error_afr_map.csv`, `afr_error_map.png`: sai lệch theo cặp.
- `trip_coverage.png`, các CSV `n_trips` và `trip_p90_p10`: độ phủ và mức khác biệt.
- Các bảng theo giai đoạn: đối chiếu khi mục tiêu thay đổi theo thời gian.
- `trip_quality.csv`, `channel_quality.csv`, `trip_cell_statistics.csv`,
  `combined_cell_statistics.csv`: truy vết từ bảng tổng hợp về từng chuyến.
- `validation_summary.csv`: kết quả kiểm chứng theo ngày.
- `aligned_trips/*.csv.gz`: 50 bảng đã xử lý riêng, gồm cột gốc, AFR quy đổi,
  thời điểm nguồn, tuổi giá trị và cờ. Đọc bằng `pd.read_csv(path, compression="gzip")`.
- `settings.json`: ngưỡng, công thức, cách gộp, giả thiết và ngày kiểm chứng.

Bảng tổng hợp cho biết những gì dữ liệu hỗ trợ. Vùng thiếu dữ liệu vẫn để trống;
chưa tạo công thức sửa nhiên liệu hoặc đề xuất mục tiêu mới cho xe.
''', 'multi-afr-export-notes'),
code('''
assert len(list((multi_output / "aligned_trips").glob("*.csv.gz"))) == 50
for metric in ["target", "measured", "error"]:
    matrix = multi_result["maps"][metric]
    saved = pd.read_csv(multi_output / f"{metric}_afr_map.csv", index_col=0)
    np.testing.assert_allclose(saved.to_numpy(), matrix.to_numpy(), equal_nan=True)
    supported_cells = multi_summary.loc[multi_summary.metric.eq(metric), "supported"].sum()
    assert matrix.notna().sum().sum() == supported_cells
    assert matrix.notna().any().any(), f"Bảng {metric} chưa có vùng đủ dữ liệu"
print("Đã kiểm tra đủ 50 chuyến, các bảng CSV khớp số liệu và giữ nguyên ô khuyết.")
print("Kết quả cuối:", multi_output.resolve())
display(multi_map_overview.round(2))
''', 'multi-afr-export'),
]
nb['cells'] = [c for c in nb['cells'] if not c.get('id', '').startswith('multi-afr-')]
nb['cells'].extend(new)
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(f'Added {len(new)} multi-trip cells to notebook.')
