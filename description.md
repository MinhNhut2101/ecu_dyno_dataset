# About Dataset

## English

### Context
This is a dataset of 50 datalogs extracted from my car's aftermarket electronic control unit (ECU). In its simplest form, the ECU of a car controls the amount of fuel and ignition timing of an internal combustion engine. The model of the ECU is Haltech Platinum Sprint 500. The car is also equipped with an Innovate MTX-L O2-wideband sensor that measures the air-to-fuel (AFR) ratio from the exhaust gas.

In combustion engines, the right amount of fuel must be injected into the engine at different conditions such as engine load and engine speed (RPM). In a ECU, the amount of fuel injected is programmed as a lookup table/map; this map is often referred to as the fuel map.

After an aftermarket ECU is first installed in a car, the car should be sent to a tuner where he can put the car on a dyno machine for tuning. However, due to cost constrains, I have left this step out and instead depends on the readings from the O2-wideband sensor to tune the fuel map myself. For gasoline engines, the stoichometric AFR ratio is 14.7. However, the car can be programmed to run leaner mixture (AFR > 14.7) or richer mixture (AFR < 14.7) for better fuel economy or higher power (for e.g., during overtaking), respectively. Also when the fuel mixture is richer, the engine runs cooler and thus enables for more ignition timing to be put on the car for more power. This is also why when a car is retro-fitted with a forced induction system such as a supercharger or turbo, the target AFR is set to richer values to cool down the engine to prevent a condition known as "engine knocking".

So, here are the ECU datalogs of my car produced from a poorly tuned fuel map. The challenge here is to produce a proper fuel map from these datalogs. By the way, the car is also not equipped with a forced induction system.

### Content
Each csv file corresponds to a trip log. The features of each log is described in the beginning of each file.

### Inspiration
The challenge here is to produce a proper 2-d fuel map from these datalogs. The axes of the fuel map is engine load and RPM.

---

## Tiếng Việt

### Bối cảnh
Đây là bộ dữ liệu gồm 50 nhật ký dữ liệu được trích xuất từ bộ điều khiển điện tử (ECU) của xe tôi. Ở dạng đơn giản nhất, ECU của ô tô kiểm soát lượng nhiên liệu và thời điểm đánh lửa của động cơ đốt trong. Mẫu ECU là Haltech Platinum Sprint 500. Xe cũng được trang bị cảm biến băng rộng Innovate MTX-L O2 đo tỷ lệ không khí trên nhiên liệu (AFR) từ khí thải.

Trong động cơ đốt trong, lượng nhiên liệu phù hợp phải được bơm vào động cơ ở các điều kiện khác nhau như tải động cơ và vòng tua (RPM). Trong ECU, lượng phun nhiên liệu được lập trình dưới dạng bảng tra cứu/bản đồ; Bản đồ này thường được gọi là bản đồ nhiên liệu.

Sau khi một ECU aftermarket được lắp đặt lần đầu vào xe, xe nên được gửi đến thợ độ để họ có thể đưa xe lên máy đo công suất để độ xe. Tuy nhiên, do hạn chế về chi phí, tôi đã bỏ qua bước này và thay vào đó dựa vào các chỉ số từ cảm biến băng rộng O2 để tự điều chỉnh bản đồ nhiên liệu. Đối với động cơ xăng, tỷ lệ AFR theo hướng dẫn là 14,7. Tuy nhiên, xe có thể được lập trình để sử dụng hỗn hợp tinh gọn hơn (AFR > 14,7) hoặc hỗn hợp đậm hơn (AFR < 14,7) để tăng tiết kiệm nhiên liệu hoặc tăng công suất (ví dụ như khi vượt mặt). Ngay cả khi hỗn hợp nhiên liệu đậm đặc, động cơ chạy mát hơn và do đó cho phép tăng thời điểm đánh lửa để tăng công suất. Đó cũng là lý do khi xe được lắp hệ thống nạp áp lực như siêu nạp hoặc turbo, AFR mục tiêu được đặt ở mức phong phú hơn để làm mát động cơ nhằm ngăn chặn tình trạng gọi là "động cơ gõ".

Đây là các bản ghi dữ liệu ECU của xe tôi được tạo ra từ bản đồ nhiên liệu được tinh chỉnh kém. Thách thức ở đây là tạo ra một bản đồ nhiên liệu chính xác từ các nhật ký dữ liệu này. Nhân tiện, xe cũng không được trang bị hệ thống tăng áp cưỡng bức.

### Nội dung
Mỗi tệp csv tương ứng với nhật ký chuyến đi. Các đặc điểm của mỗi nhật ký được mô tả ở đầu mỗi tệp.

### Cảm hứng
Thách thức ở đây là tạo ra một bản đồ nhiên liệu 2 chiều đúng chuẩn từ các nhật ký dữ liệu này. Các trục của bản đồ nhiên liệu là tải động cơ và vòng tua máy.

---

