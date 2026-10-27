from pathlib import Path
import json
import shutil
import textwrap

root = Path(__file__).resolve().parents[1]
path = root / 'main_copy.ipynb'
backup = root / '_notebook_work/main_copy.before_linear_completion.ipynb'
if not backup.exists():
    shutil.copy2(path, backup)
nb = json.loads(path.read_text(encoding='utf-8'))

def cell(kind, text, name):
    item = {'cell_type': kind, 'id': name, 'metadata': {},
            'source': textwrap.dedent(text).strip().splitlines(keepends=True)}
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    return item

def md(text, name):
    return cell('markdown', text, name)

def code(text, name):
    return cell('code', text, name)

new = [
md(r'''
## 11. Nội suy tuyến tính hai chiều và bảng AFR ước lượng đầy đủ

Giữ nguyên bảng có dữ liệu hỗ trợ ở phần 10. Phần này tạo hai sản phẩm riêng:

1. **Nội suy thuần**: giữ nguyên ô đủ dữ liệu và điền các ô trong miền bao quanh
   các điểm nguồn. Ngoài miền giữ NaN, đúng hành vi của `LinearNDInterpolator`.
2. **Bảng đầy đủ để thực hành**: bổ sung **ngoại suy tuyến tính** cho vùng ngoài
   miền. Mỗi ô được gắn nguồn `supported`, `linear_interpolation` hoặc
   `linear_extrapolation`. Bảng đầy đủ không được gọi là toàn bộ số đo thực.

Với bảng Target AFR hiện tại: 81 ô đủ dữ liệu, nội suy chỉ thêm được 1 ô;
238 ô còn lại là ngoại suy. Việc tạo đủ 320 ô không làm dữ liệu thực phong phú hơn.
Các giai đoạn được dựng riêng để tránh chỉ có một bảng gộp các mục tiêu thay đổi.

Hàm tính nằm trong `afr_interpolation.py`. Chỉ đọc kết quả đã lưu của phần 10;
không cần xử lý lại 50 CSV khi thử nội suy trên cùng kết quả nguồn.

Nguồn: [SciPy LinearNDInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html)
và [giới hạn của ngoại suy](https://docs.scipy.org/doc/scipy/tutorial/interpolate/extrapolation_examples.html).
''', 'linear-afr-intro'),
code('''
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from afr_interpolation import run_completion, plot_completion

LINEAR_NEIGHBORS = 8
linear_result = run_completion("outputs/all_trips_afr", neighbors=LINEAR_NEIGHBORS)
linear_output = linear_result["output_dir"]
display(linear_result["coverage"].round(4))
print("Kết quả:", linear_output.resolve())
''', 'linear-afr-run'),
md(r'''
### 11.1. Công thức và bảo toàn điểm nguồn

Trong từng tam giác của miền nguồn, dùng tổ hợp tuyến tính:

$$\widehat{AFR}(x,y)=w_1AFR_1+w_2AFR_2+w_3AFR_3,\quad \sum w_i=1.$$

Chỉ các ô đủ hỗ trợ trong phần 10 làm điểm nguồn. Không lấy các ô vừa sinh ra
làm nguồn cho bước tiếp theo, không sửa các ô nguồn.

Để điền vùng ngoài, khớp mặt phẳng cục bộ với các điểm nguồn gần nhất:

$$AFR=a+b(x-x_q)+c(y-y_q),\qquad \widehat{AFR}(x_q,y_q)=a.$$

Tọa độ tính theo đơn vị khoảng: một khoảng RPM=500 và một khoảng Load=50 đều
có độ dài 1. Trọng số khớp là $1/(1+d^2)$. Bắt đầu từ 8 điểm; tăng số điểm nếu
thiết kế không đủ hạng 3 hoặc điều kiện số quá xấu. Không tự thay bằng phương pháp
láng giềng gần nhất, không kẹp giá trị dự đoán vào một khoảng đẹp mắt.
Khoảng cách đến nguồn được lưu để chỉ ra vùng đi xa khỏi dữ liệu, không diễn giải
nó thành xác suất tin cậy. Bảng sai lệch được khớp từ sai lệch cặp ở phần 10,
không bắt buộc bằng hiệu của hai bề mặt AFR được khớp độc lập.
''', 'linear-afr-formula-notes'),
code('''
linear_target = linear_result["results"]["all_50_trips_target"]
linear_measured = linear_result["results"]["all_50_trips_measured"]
linear_target_full = linear_target["full"]
linear_target_strict = linear_target["strict"]
linear_target_origin = pd.DataFrame(
    linear_target["audit"].origin.to_numpy().reshape(linear_target_full.shape),
    index=linear_target_full.index, columns=linear_target_full.columns,
)
print("Mã hàng/cột bên dưới là số khoảng; CSV và hình dùng khoảng RPM/Load đầy đủ.")
print("Bảng chỉ nội suy, phần ngoài miền vẫn trống:")
display(linear_target_strict.round(2))
print("Bảng đầy đủ có ngoại suy, phải đọc cùng bảng nguồn của từng ô:")
display(linear_target_full.round(2))
display(linear_target["audit"].groupby("origin").agg(
    cells=("origin", "size"),
    max_distance_bins=("nearest_anchor_distance_bins", "max"),
    max_neighbors=("neighbors_used", "max"),
).round(3))
''', 'linear-afr-values'),
md('''
### 11.2. Bảng màu và nguồn của các ô được điền

Không ký hiệu: ô đủ dữ liệu; **I**: nội suy; **E**: ngoại suy. Bảng màu nguồn
được xuất riêng để không che mất việc phần lớn bảng đang là ước lượng ngoài miền.
Màu AFR tự mở rộng nếu cần; số trong ô được giữ đúng, không kẹp hay xóa dự đoán
chỉ vì nằm ngoài khoảng giá trị nguồn. Những trường hợp đó có cờ trong báo cáo.
''', 'linear-afr-plot-notes'),
code('''
linear_figures = plot_completion(linear_result)
plt.show()
''', 'linear-afr-plots'),
md('''
### 11.3. Kiểm chứng theo dải RPM và dải Load được giấu đi

Lần lượt giấu nguyên một cột RPM, khớp lại từ các cột còn lại rồi dự đoán cột bị
giấu. Làm tương tự với hàng Load. Mỗi điểm kiểm chứng không xuất hiện trong nguồn
khớp của chính lần kiểm chứng đó. Báo cáo riêng nội suy và ngoại suy vì hai bài
toán có độ khó khác nhau; không gộp thành một chỉ số dễ gây hiểu nhầm.

Các kiểm tra này chỉ đánh giá gần vùng đã biết. Chúng chưa kiểm chứng các vùng
RPM cao hoặc tải rất xa nguồn. Chúng bổ sung cho kiểm chứng giữ riêng ngày/chuyến
ở phần 10, không thay thế bước đó và không chứng minh bảng mục tiêu đã tối ưu.
''', 'linear-afr-validation-notes'),
code('''
linear_validation = linear_result["validation"]
display(linear_validation.loc[linear_validation.dataset.eq("all_50_trips")].round(4))
print("Kiểm tra theo giai đoạn:")
display(linear_validation.loc[~linear_validation.dataset.eq("all_50_trips")].round(4))
linear_range_flags = linear_result["coverage"][[
    "dataset", "metric", "full_min", "full_max", "far_extrapolated_cells",
    "outside_anchor_value_range_cells", "nonpositive_cells"
]]
display(linear_range_flags.round(4))
''', 'linear-afr-validation'),
md('''
### 11.4. Đọc các bảng đã xuất

Trong `outputs/all_trips_afr/linear_completion/`:

- `all_50_trips_target_full.csv` và `.png`: bảng Target AFR đủ ô, có phần ngoại suy.
- `all_50_trips_target_strict.csv`: bảng chỉ nội suy tuyến tính trong miền nguồn.
- `all_50_trips_target_cell_origin.csv`: nguồn của từng ô.
- `all_50_trips_target_cell_audit.csv`: khoảng cách, số điểm nguồn, hạng và cờ kiểm tra.
- Các file `measured` và `error`: xử lý tương tự cho cảm biến và sai lệch.
- Các file có tên giai đoạn: bảng Target AFR riêng từng giai đoạn.
- `completion_summary.csv`, `spatial_validation.csv`, `settings.json`: số lượng,
  kết quả kiểm chứng và các giả thiết tính toán.

Ô có quan sát thưa trong bảng nguồn chưa đủ ngưỡng không được tự nâng thành ô
đủ dữ liệu. `source_cell_supporting_trips` chỉ mô tả dữ liệu gốc tại vị trí đó;
điểm được nội/ngoại suy không được nhận giả số chuyến hỗ trợ.
''', 'linear-afr-export-notes'),
code('''
for key, result in linear_result["results"].items():
    source = result["source"].to_numpy()
    full = result["full"].to_numpy()
    strict = result["strict"].to_numpy()
    known = np.isfinite(source)
    np.testing.assert_array_equal(source[known], full[known])
    np.testing.assert_array_equal(source[known], strict[known])
    assert np.isfinite(full).all(), f"{key}: chưa đủ ô hữu hạn"
    audit = result["audit"]
    exterior = audit.origin.eq("linear_extrapolation").to_numpy().reshape(full.shape)
    assert np.isnan(strict[exterior]).all()
    assert audit.origin.isin(["supported", "linear_interpolation", "linear_extrapolation"]).all()
    saved = pd.read_csv(linear_output / f"{key}_full.csv", index_col=0)
    np.testing.assert_allclose(saved.to_numpy(), full)
    assert len(audit) == source.size
print("Đã kiểm tra: đủ ô; bảo toàn nguồn; nội suy và ngoại suy được tách rõ; CSV khớp kết quả.")
display(linear_result["coverage"].round(3))
''', 'linear-afr-export'),
]
nb['cells'] = [c for c in nb['cells'] if not c.get('id', '').startswith('linear-afr-')]
nb['cells'].extend(new)
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(f'Added {len(new)} cells for linear completion.')
