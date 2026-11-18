# Báo cáo Day 09 — Phan Hải Đăng, 2A202603229

## 1. Kết luận trước
Pipeline kết luận **HOLD** (mã thoát `10`). Từ **224** ảnh thô ban đầu, bước `intake` loại **22** ảnh lỗi/trùng/rò rỉ, nhận **202/224** ảnh vào xử lý; sau `autofix` (35 thay đổi) và mô phỏng người duyệt (80 ảnh), còn **0** lỗi bắt buộc (`ERROR`) nhưng vẫn còn **38/202** ảnh (`18.81% >= 10%`) mang cảnh báo chưa qua mắt người nên xuất bản **nháp** `submission/export/clean_v1/` và chưa phát hành chính thức.

## 2. Lỗi và phân việc
Số liệu trích từ `submission/reports/02_error_list_v1.csv` (trước khi sửa):

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| `DUP_IMAGE_EXACT` | 8 | Máy (`remove_image`) | Trùng byte hoàn toàn (md5), xóa dư thừa an toàn 100%. |
| `LEAK_TEST` | 5 | Máy (`remove_image`) | Trùng với tập `test`/`gold_val`, phải loại để tránh metric đẹp giả. |
| `DUP_IMAGE_NEAR` | 4 | Máy (`remove_image`) | Trùng gần đúng (`phash_dist <= 10`), xóa bản sao an toàn. |
| `BAD_SIZE` | 3 | Máy (`remove_image`) | Cạnh ngắn `< 250px`, không đạt chuẩn phân giải dự án. |
| `CORRUPT_IMAGE` | 2 | Máy (`remove_image`) | Ảnh hỏng không đọc được bằng `PIL.Image.load()`. |
| `BOX_OUT_OF_IMAGE` | 8 | Máy (`clip_box`) | Tràn mép `[0, 1]`, kẹp tọa độ về biên ảnh được an toàn. |
| `DUP_BOX` | 7 | Máy (`drop_line`) | Trùng box cùng lớp (`IoU > 0.9`), xóa dòng lặp. |
| `BAD_CLASS_ID` | 6 | Máy (`drop_line`) | ID lớp ngoài `0..3`, máy không tự đoán lớp đúng nên xóa dòng lỗi. |
| `BOX_ZERO` | 4 | Máy (`drop_line`) | `w <= 0` hoặc `h <= 0`, box vô nghĩa nên xóa dòng. |
| `BAD_FORMAT` | 3 | Máy (`drop_line`) | Sai 5 cột số YOLO (`NaN`/thiếu cột), xóa dòng hỏng. |
| `BAD_CLASS_NAME` | 2 | Máy (`rename_classes`) | Sai tên/thứ tự lớp trong `data.yaml`, chuẩn hóa theo `cfg["classes"]`. |
| `MISSING_LABEL_FILE` | 6 | **Người** (`needs_label`) | Máy không được tự bịa nhãn hay tạo file rỗng; đẩy lên đầu hàng đợi cho người gán. |
| `POSSIBLE_MISSING` | 53 | **Người** (`WARN`) | Nghi thiếu nhãn dựa trên `yolo26l`, cần mắt người xác nhận. |
| `POSSIBLE_LOOSE_BOX` | 28 | **Người** (`WARN`) | Box nghi lỏng (`IoU < 0.5`), cần người căn chỉnh lại. |
| `POSSIBLE_FALSE_POS` | 25 | **Người** (`WARN`) | Box không khớp dự đoán nào của model, nghi gán thừa. |
| `POSSIBLE_WRONG_CLASS` | 18 | **Người** (`WARN`) | Nghi nhầm lớp (`IoU >= 0.5` khác lớp), cần người phân biệt ngữ nghĩa. |
| `EMPTY_LABEL` | 12 | **Người** (`WARN`) | File nhãn rỗng (ảnh nền thật hay bị sót nhãn?), cần người kiểm. |
| `BOX_TINY` | 6 | **Người** (`WARN`) | Cạnh ngắn `< 12px`, cần người xem có nên giữ theo guideline. |
| `CLASS_RARE` | 4 | **Người** (`WARN`) | Lớp ít mẫu (`< 5` box) ở mức dataset. |
| `BOX_HUGE` | 3 | **Người** (`WARN`) | Box chiếm `> 95%` diện tích ảnh, cần người kiểm tra. |

`05_review_queue.csv` xếp các ảnh như `img_0020` (`priority=11.524`, dính `FIXED_DROP_LINE;POSSIBLE_MISSING;POSSIBLE_WRONG_CLASS`) và 6 ảnh `NEEDS_LABEL` (`priority=10.0`: `img_0031`, `img_0053`, `img_0126`, `img_0135`, `img_0170`, `img_0219`) lên đầu vì điểm ưu tiên là tổng trọng số rủi ro (`NEEDS_LABEL=10`, `POSSIBLE_WRONG_CLASS=3`, `POSSIBLE_MISSING=2`, `FIXED_DROP_LINE=2`, cộng thêm `conf` của model). Với ngân sách **80/125** ảnh trong hàng đợi, người duyệt đã xử lý được toàn bộ **6** ảnh thiếu file nhãn và **74** ảnh cảnh báo ngữ nghĩa nặng nhất (giảm số cảnh báo `second_opinion` từ `124` xuống còn `57` ở `07_error_list_v3.csv`).

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | **10** | `DUP_IMAGE_NEAR`: `1.00 / 0.50` → `1.00 / 0.67` (`F1: 0.67 → 0.80`) | `remove_image` xóa thẳng ảnh nên ưu tiên `Precision = 1.00`. Chọn `10` để bắt thêm ảnh resize/nén mà không gộp nhầm ảnh giao thông cùng góc; tránh `20` vì làm `DUP_IMAGE_EXACT` tụt recall xuống `0.33`. |
| `second_opinion.conf` | 0.25 | **0.6** | `WRONG_CLASS`: `0.56 / 0.83` → `0.62 / 0.83`<br/>`MISSING`: `0.50 / 1.00` → `0.57 / 1.00` | Giữ nguyên `Recall = 1.00` cho `POSSIBLE_MISSING` (không bỏ sót vật thiếu nhãn), đồng thời giảm báo động giả để tiết kiệm ngân sách duyệt 80 ảnh. Không chọn `0.7–0.8` vì `MISSING` tụt recall xuống `0.88–0.75`. |

**Chỗ calibration có thể đánh lừa:** Tập `calib` chỉ có **39 ảnh** (mỗi loại lỗi 2–8 ca nên Precision/Recall nhảy bước 15–50%), được dựng từ nhãn người sạch và **không có** `LEAK_TEST`, trong khi `raw` là nhãn model (nhiều lỗi tự nhiên khác biệt). Trên `calib`, `phash_dist = 16` đạt `F1 = 0.91` (`5/6` ca gần trùng), nhưng áp sang `raw` có thể xóa nhầm ảnh khác nhau có bố cục đường phố tương đồng; và 1 ca biến đổi nặng vẫn không ngưỡng nào bắt được.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
- **(a) Chứng minh Gate chặn `Rework`:** Chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` (không có người duyệt) cho kết quả **`REWORK — còn 6 lỗi bắt buộc (ERROR)`** (chính là 6 ảnh `MISSING_LABEL_FILE`), mã thoát **`20`**, và **không tạo** thư mục `work/exp_nb`.
- **(b) Điểm mù pipeline không bắt được (từ soi 10 ảnh `work/viz/`):**
  1. `img_0001` (thiếu nhãn 1 người lính) và `img_0030` (nhiều `person` chưa gán): khi vật bị che khuất hoặc nhỏ khiến `yolo26l` cũng có `conf < 0.6` hoặc diện tích `< 600 px²`, `second_opinion` sẽ im lặng.
  2. `img_0002` (vật mờ bên phải gán thành `person`) và `img_0005` (object 5 gán `person` không rõ): khi nhãn gốc trùng với dự đoán sai của chính `yolo26l` hoặc `IoU` nằm vùng mơ hồ, máy không thể tự phân biệt đúng sai ngữ nghĩa.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Cấu hình: `yolo26n`, `20` epochs, `2` seeds (`0`, `1`) trên Google Colab T4 GPU.

| biến thể | #ảnh train | số seed | map50 (TB ± độ lệch) | map50_95 (TB ± độ lệch) | precision (TB ± độ lệch) | recall (TB ± độ lệch) |
|---|---|---|---|---|---|---|
| raw | 224 | 2 | 0.726 ± 0.002 | 0.507 ± 0.001 | 0.759 ± 0.049 | 0.632 ± 0.014 |
| clean | 202 | 2 | 0.703 ± 0.027 | 0.486 ± 0.017 | 0.768 ± 0.007 | 0.631 ± 0.014 |

* **Chênh `mAP50 (clean − raw)` = `-0.023`** (trong khi độ lệch chuẩn giữa 2 seed của `clean` là `±0.027`).
* **Kết luận trung thực:** Chênh lệch `-0.023` **nằm trong độ nhiễu giữa các seed (`0.027`)**, nên **chưa đủ cơ sở kết luận** `clean` tốt hơn hay kém hơn `raw` về `mAP50` (dù `precision` của `clean` nhỉnh hơn `0.768` vs `0.759` và ổn định hơn giữa các seed: `std 0.007` vs `0.049`).
* **Hai giả thuyết giải thích & cách kiểm:**
  1. *Metric đẹp giả do rò rỉ test (`LEAK_TEST`):* `raw` chứa **5 ảnh trùng tập `test`/`gold_val`** và **12 ảnh trùng**, giúp `raw` học thuộc một phần ảnh đánh giá; `clean` đã loại 5 ảnh rò rỉ này nên điểm test thấp hơn nhưng trung thực hơn. **Cách kiểm:** Đo lại mAP sau khi loại 5 ảnh rò rỉ khỏi mẫu đánh giá hoặc kiểm tra trên tập test độc lập hoàn toàn.
  2. *Nhiễu cỡ mẫu nhỏ & `clean` đang ở trạng thái `HOLD`:* Tập `clean` giảm từ 224 xuống 202 ảnh (~10%), vẫn còn **38/202 ảnh cảnh báo chưa duyệt**, lại chỉ train `20` epoch với `2` seed trên tập test `100` ảnh nên phương sai `clean` lớn (`std = 0.027`). **Cách kiểm:** Duyệt thêm để đạt `Release`, rồi chạy lại với `>= 5` seed và `50` epoch.

## 6. Quyết định và ghi chú AI
**(a) Quyết định cuối:** Hiện giữ trạng thái **HOLD** (`38/202 = 18.81% >= 10%`), xuất bản nháp `clean_v1/` để thử nghiệm train. Để đạt **Release**, cần tăng ngân sách duyệt thêm ít nhất **18–20 ảnh** (từ 80 lên ~100 ảnh) để kéo số ảnh còn cảnh báo chưa duyệt xuống dưới `20/202` (`< 10%`).

**(b0) Tìm hiểu kiến trúc & 4 module (CP2):**
- **Prompt →** Yêu cầu phân tích kiến trúc pipeline AutoQC, vị trí, chức năng, vào/ra và bẫy của 4 module (`label_format.py`, `gate.py`, `autofix.py`, `second_opinion.py`).
- **AI trả gì →** Sơ đồ luồng `intake -> QC (3 vòng) -> autofix -> human review -> gate -> export` và bảng đặc tả vào/ra từng module.
- **Tôi kiểm bằng cách nào →** Đối chiếu trực tiếp với `docs/PIPELINE.md`, `pipeline/run.py`, và docstring của 4 file stub.
- **Chỗ AI sai/thiếu và tôi sửa thế nào →** Ban đầu AI tóm tắt chung; tôi yêu cầu chỉ rõ vị trí gọi hàm trong `run.py` và danh sách bẫy từng module trước khi code.

**(b1) `label_format.py`:**
- **Prompt →** Viết `pipeline/checks/label_format.py` nhận `ds`, `cfg`; kiểm tra từng dòng YOLO, trả `list[Issue]` đủ 7 mã (`BAD_FORMAT`, `BAD_CLASS_ID`, `BOX_ZERO`, `BOX_OUT_OF_IMAGE` kèm `extra={"overflow": ...}`, `DUP_BOX`, `BOX_TINY`, `BOX_HUGE`), giữ số dòng gốc khi có dòng trắng, đọc ngưỡng từ `cfg["label_format"]`.
- **AI trả gì →** Hàm `run(ds, cfg)` dùng `ds.read_label_lines(stem)`, `parse_yolo_line`, `iou_xywh`, `ds.image_size(stem)` và kiểm tra đủ 7 mã.
- **Tôi kiểm bằng cách nào →** Đọc từng dòng (không xóa/ghi file, không `except: pass`, không hardcode ngưỡng) và chạy `python -m pytest tests/test_label_format.py -q` (`14 passed`).
- **Chỗ AI sai/thiếu và tôi sửa thế nào →** Lần đầu cài môi trường trên Python 3.15.2, `pip install -r requirements.txt` lỗi build wheel `pyyaml` nên phải cài thẳng `pip install imagehash`; trong code yêu cầu kiểm tra `img_size is not None` trước khi tính `BOX_TINY` để không crash với nhãn mồ côi.

**(b2) `gate.py`:**
- **Prompt →** Viết `pipeline/steps/gate.py` hàm `decide(issues, image_stems, cfg, reviewed=frozenset())` trả `dict` (`decision`, `n_errors`, `n_warn_images`, `n_images`, `warn_ratio`, `reasons`), `ERROR` không bị `reviewed` bỏ qua, tránh chia cho `0`.
- **AI trả gì →** Hàm `decide` đếm `n_errors` toàn bộ `ERROR`, dùng `set` đếm ảnh `WARN` duy nhất trong `image_stems` và ngoài `reviewed`, xét ngưỡng `>= cfg["gate"]["warn_ratio_hold"]`.
- **Tôi kiểm bằng cách nào →** Kiểm tra không hardcode `0.10`, chạy `python -m pytest tests/test_gate.py -q` (`10 passed`) và chạy `pipeline.run --budget 0` ra `Rework` (exit `20`).
- **Chỗ AI sai/thiếu và tôi sửa thế nào →** Yêu cầu làm rõ việc lọc bỏ issue cấp dataset (`"<dataset>"`, `"data.yaml"`) và ảnh đã xóa khỏi `image_stems` để không làm sai mẫu số/tử số.

**(b3) `autofix.py`:**
- **Prompt →** Viết `pipeline/steps/autofix.py` hàm `fix(ds, issues, cfg) -> list[dict]` chỉ sửa `ERROR` trên `ds` (`remove_image`, `rename_classes`, `needs_label` không tạo file, `drop_line`, `clip_box`), gom theo file giữ số dòng gốc, `drop_line` thắng `clip_box`.
- **AI trả gì →** Hàm `fix` + helper `_clip_yolo_box` xử lý đủ 5 nhóm action và ghi log trước–sau.
- **Tôi kiểm bằng cách nào →** Kiểm tra không sửa `WARN`, không đụng `data/raw/`, chạy `python -m pytest tests/test_autofix.py -q` (`11 passed`).
- **Chỗ AI sai/thiếu và tôi sửa thế nào →** Yêu cầu xử lý rõ ca box nằm hẳn ngoài ảnh sau clip (`x2 <= x1` hoặc `y2 <= y1`) phải chuyển thành `drop_line` thay vì ghi box `0 0`, và bỏ qua dòng nhãn của ảnh vừa bị `remove_image`.

**(b4) `second_opinion.py`:**
- **Prompt →** Viết `pipeline/checks/second_opinion.py` hàm `run(ds, cfg, preds=None)` so nhãn với dự đoán `yolo26l` sinh 4 `WARN` (`POSSIBLE_MISSING`, `POSSIBLE_LOOSE_BOX`, `POSSIBLE_WRONG_CLASS`, `POSSIBLE_FALSE_POS`), giữ số dòng gốc, đọc 6 ngưỡng từ `cfg["second_opinion"]`.
- **AI trả gì →** Hàm `run` 2 pha (Pha A lọc `conf` + `min_area_px`, Pha B lọc `fp_conf` không lọc diện tích) + helper `_read_valid_labels`.
- **Tôi kiểm bằng cách nào →** Chạy `python -m pytest tests/test_second_opinion.py -q` (`15 passed`) và toàn bộ `pytest -q` (`79 passed`).
- **Chỗ AI sai/thiếu và tôi sửa thế nào →** Yêu cầu dùng `ds.read_label_lines` thay cho `ds.valid_boxes` (vì `valid_boxes` mất số dòng gốc khi có dòng trắng/hỏng) và im lặng ở ca khác lớp có IoU mơ hồ `[iou_missing, iou_match)`.
