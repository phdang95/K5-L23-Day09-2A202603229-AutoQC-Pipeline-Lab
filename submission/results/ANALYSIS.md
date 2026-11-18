# Phân tích kết quả Train: `raw` vs `clean`

## 1. Chênh `mAP50(clean) − mAP50(raw)` so với độ lệch chuẩn giữa các seed
- `raw` (224 ảnh, 2 seed): `mAP50 = 0.726 ± 0.002` (`0.7248`, `0.7274`).
- `clean` (202 ảnh, 2 seed): `mAP50 = 0.703 ± 0.027` (`0.7223`, `0.6846`).
- Chênh `mAP50(clean) − mAP50(raw) = -0.023`, nhỏ hơn độ lệch chuẩn giữa 2 seed của `clean` (`0.027`).
- **Kết luận:** Với chỉ 2 seed và 20 epoch trên tập test 100 ảnh, chênh lệch nằm gọn trong nhiễu ngẫu nhiên — **chưa đủ bằng chứng** kết luận `clean` kém hơn `raw` (ngược lại, `precision` của `clean` tăng nhẹ `0.759 → 0.768` và ổn định hơn: `std 0.007` vs `0.049`).

## 2. `raw` thật sự train trên bao nhiêu ảnh?
- Theo cấu hình danh mục, `raw` có **224** ảnh (`train_images: 224`), nhưng khi Ultralytics quét dữ liệu sẽ **tự động bỏ qua** các ảnh không đọc được (`CORRUPT_IMAGE`: 2 ảnh), ảnh thiếu nhãn (`MISSING_LABEL_FILE`: 6 ảnh coi như background/bỏ qua tùy cấu hình), và các dòng nhãn hỏng (`BAD_FORMAT`: 3, `BAD_CLASS_ID`: 6, `BOX_ZERO`: 4, `BOX_OUT_OF_IMAGE`: 8). Vì vậy số ảnh/nhãn thực sự góp gradient sạch vào model ít hơn 224.

## 3. `raw` có ảnh trùng với tập test không?
- **Có.** Bước `intake` phát hiện **`LEAK_TEST = 5` ảnh** trong `raw` trùng với tập `test`/`gold_val` (cùng `12` ảnh trùng nội bộ `8 exact + 4 near`).
- **Tác động:** 5 ảnh rò rỉ làm `raw` "học vẹt" một phần tập đánh giá, thổi phồng metric giả tạo trên `test`; `clean` đã loại 5 ảnh này nên điểm phản ánh đúng năng lực tổng quát hóa hơn.
- **Cách kiểm:** Loại 5 ảnh rò rỉ khỏi tập đánh giá (hoặc đo trên tập test độc lập hoàn toàn) rồi so lại `mAP50` của hai model.

## 4. Lớp nào (`car` / `person` / `truck` / `bus`) được lợi nhất?
Tính trung bình `AP50` qua 2 seed:
- **`bus`:** `0.6996 → 0.7137` (**`+0.0141`**) — tăng mạnh ở seed 0 (`0.6833 → 0.7401`).
- **`person`:** `0.8213 → 0.8271` (**`+0.0058`**) — tăng ổn định ở cả 2 seed (`0.8122 → 0.8228` và `0.8304 → 0.8314`).
- **`car`:** `0.7395 → 0.7401` (`+0.0006`, gần như giữ nguyên).
- **`truck`:** `0.6440 → 0.5329` (`-0.1111`, kéo tụt điểm chung của `clean`).
- **Giải thích từ `02_error_list_v1.csv`:** Lớp **`bus`** và **`person`** được lợi nhất vì `autofix` + người duyệt đã sửa các ca `BAD_CLASS_ID`, `BOX_OUT_OF_IMAGE`, `POSSIBLE_WRONG_CLASS` và bổ sung nhãn thiếu (`POSSIBLE_MISSING`). Ngược lại, `truck` vốn là lớp ít mẫu (`CLASS_RARE`) lại dễ bị nhiễu khi tập `clean` vẫn còn 38 ảnh chưa duyệt và mất 22 ảnh so với `raw`.

## 5. `clean` còn lỗi gì? Đây đã là chặn trên của "dữ liệu sạch" chưa?
- Theo `07_error_list_v3.csv`, `clean` đã sạch `0` lỗi bắt buộc (`ERROR`), nhưng hàng đợi `05_review_queue.csv` có **125 ảnh > ngân sách 80 ảnh**, nên vẫn còn **38/202 ảnh (`18.81%`)** mang cảnh báo ngữ nghĩa chưa duyệt (`POSSIBLE_FALSE_POS: 21`, `POSSIBLE_LOOSE_BOX: 20`, `POSSIBLE_MISSING: 13`, `BOX_TINY: 7`, `EMPTY_LABEL: 4`, `BOX_HUGE: 3`, `POSSIBLE_WRONG_CLASS: 3`) $\implies$ Gate kết luận **`HOLD`**.
- **Chưa phải chặn trên ("oracle"):** Bản `clean_v1` mới chỉ là bản nháp sau 1 vòng lọc + 80 ảnh duyệt; muốn chạm chặn trên của dữ liệu sạch cần duyệt hết 38 ảnh còn lại để đạt **`Release`** (`< 10%` cảnh báo).
