# Báo cáo Day 09 — <Họ và tên>, <MSHV>

> Xoá mọi `TODO(học viên)` và dòng gợi ý. Mỗi ý trả lời bằng **số lấy từ `submission/reports/`**, kèm mẫu số. Cả báo cáo ~1–2 trang.

## 1. Kết luận trước
TODO(học viên): pipeline kết luận **Release / Hold / Rework**? Bao nhiêu ảnh vào, bao nhiêu ảnh ra dataset? Lý do chính (1–3 câu).

## 2. Lỗi và phân việc
Lấy từ `02_error_list_v1` (gom theo mã lỗi). Cột "ai sửa": **máy** (autofix) hay **người** (hàng đợi); chỉ ghi những mã có mặt trong lô của bạn.

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| TODO(học viên) | | | |

TODO(học viên): vì sao `05_review_queue.csv` xếp ảnh đầu hàng như vậy? Với ngân sách 80 ảnh, người xem được bao nhiêu lỗi ngữ nghĩa?

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | TODO(học viên) | | |
| `second_opinion.conf` | 0.25 | TODO(học viên) | | |

TODO(học viên): **một chỗ calibration có thể đánh lừa bạn**, kèm bằng chứng từ dữ liệu của bạn (vd. mở xem vài cặp ảnh bị cho là trùng; số ảnh bị loại ở bước 1 có hợp lý không?).

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
TODO(học viên): (a) chứng minh gate chặn đúng: chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` (không có người duyệt nên ảnh thiếu file nhãn còn nguyên), chỉ ra kết luận **Rework**, mã thoát 20 và việc **không xuất** dataset. (Đừng sửa tay `work/run/ds`: pipeline dựng lại thư mục này từ raw mỗi lần chạy.) (b) ≥ 2 điểm mù có bằng chứng (box lỏng nhẹ, nhầm lớp mà model lớn cũng nhầm, ảnh trùng biến đổi mạnh…).

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Chưa train: ghi "chưa chạy" và lý do, rồi bỏ qua phần còn lại của mục này. Nếu đã train: dán bảng từ `submission/results/comparison.md` (model, epoch, seed, số ảnh train mỗi biến thể).

TODO(học viên): chênh mAP50 (clean − raw) so với **độ lệch chuẩn giữa các seed**; kết luận **trung thực** (nằm trong nhiễu thì nói là chưa kết luận được); ≥ 2 giả thuyết giải thích (ảnh trùng/rò rỉ test trong `raw`, nhãn máy còn lại trong `clean`, test chỉ 100 ảnh…) và cách kiểm.

## 6. Quyết định và ghi chú AI
TODO(học viên): (a) quyết định cuối; nếu Hold/Rework cần làm gì để thành Release?

(b0) Prompt tìm hiểu kiến trúc & 4 module (CP2):
- **Prompt tìm hiểu:** Phân tích kiến trúc pipeline AutoQC và xác định rõ vị trí, chức năng, đầu vào, đầu ra, cùng các ca bẫy của 4 module cần viết (`label_format.py`, `gate.py`, `autofix.py`, `second_opinion.py`) trước khi cài đặt TDD.
- **Tóm tắt vị trí & vào/ra:** `label_format.py` + `second_opinion.py` (chạy trong bộ QC 3 vòng: nhận `ds` + `cfg` $\to$ trả `list[Issue]`); `autofix.py` (chạy giữa QC v1 và v2: nhận `ds` + `issues` `ERROR` $\to$ sửa `ds`, trả `list[dict]` ghi `03_fixes.csv`); `gate.py` (chốt sau QC v3: nhận `issues3` + `stems` + `reviewed` $\to$ trả `verdict` quyết định `Rework`/`Hold`/`Release` & exit code `20`/`10`/`0`).

(b) Nhật ký dùng AI (`label_format.py`):
- **Prompt chính:** Viết `pipeline/checks/label_format.py` (Python 3) nhận đối tượng `ds` (từ `work/run/ds/`, không sửa dữ liệu gốc) và `cfg`; đọc từng dòng YOLO (`id cx cy w h`), trả về `list[Issue]` phân biệt đủ 7 mã (`BAD_FORMAT`, `BAD_CLASS_ID`, `BOX_ZERO`, `BOX_OUT_OF_IMAGE` kèm `extra={"overflow": ...}`, `DUP_BOX`, `BOX_TINY`, `BOX_HUGE`), giữ nguyên số dòng khi có dòng trống, đọc mọi ngưỡng từ `cfg["label_format"]`, giải thích từng đoạn bằng tiếng Việt.
- **Cách kiểm:** Đọc kiểm tra từng dòng (không xóa/ghi file, không `except: pass`, không hardcode ngưỡng, có kiểm tra `img_size is not None`), chạy `python -m pytest tests/test_label_format.py -q` đạt `14 passed`.
- **Lỗi gặp & cách xử lý:** Khi cài môi trường trên Python 3.15.2, `pip install -r requirements.txt` lỗi build wheel `pyyaml` (thiếu MSVC Build Tools) nên cài trực tiếp `pip install imagehash`; trong `label_format.py` kiểm tra kỹ `img_size is not None` trước khi tính `BOX_TINY` để tránh crash với nhãn không có ảnh.

(b2) Nhật ký dùng AI (`gate.py`):
- **Prompt chính:** Viết `pipeline/steps/gate.py` hàm `decide(issues, image_stems, cfg, reviewed=frozenset())` trả về `dict` (`decision`, `n_errors`, `n_warn_images`, `n_images`, `warn_ratio`, `reasons`); đếm `ERROR` không loại theo `reviewed`, đếm ảnh `WARN` duy nhất nằm trong `image_stems` và ngoài `reviewed`, tránh chia cho `0`, so `>= cfg["gate"]["warn_ratio_hold"]`.
- **Cách kiểm:** Kiểm tra không hardcode ngưỡng `0.10`, không bỏ qua `ERROR` đã duyệt, chạy `python -m pytest tests/test_gate.py -q` đạt `10 passed`.
- **Lưu ý / bẫy đã chặn:** Xử lý biên `n_images == 0` trả `warn_ratio = 0.0`, lọc bỏ issue cấp dataset (`"<dataset>"`, `"data.yaml"`) hoặc ảnh đã xóa không thuộc `image_stems`.

(b3) Nhật ký dùng AI (`autofix.py`):
- **Prompt chính:** Viết `pipeline/steps/autofix.py` hàm `fix(ds, issues, cfg) -> list[dict]` chỉ sửa lỗi `ERROR` trên bản sao `ds` (`remove_image`, `rename_classes`, `needs_label` không tạo file, `drop_line`, `clip_box` hoặc `drop_line` nếu box ngoài ảnh hoàn toàn), gom theo file giữ nguyên số dòng gốc, ưu tiên `drop_line` trước `clip_box`.
- **Cách kiểm:** Kiểm tra không sửa `WARN`, không đụng `data/raw/`, chạy `python -m pytest tests/test_autofix.py -q` đạt `11 passed`.
- **Lưu ý / bẫy đã chặn:** Xử lý box nằm hẳn ngoài ảnh sau clip (`x2 <= x1` hoặc `y2 <= y1`) chuyển thành `drop_line` thay vì ghi box `0 0`, và bỏ qua issue dòng nhãn của ảnh vừa bị `remove_image`.

(b4) Nhật ký dùng AI (`second_opinion.py`):
- **Prompt chính:** Viết `pipeline/checks/second_opinion.py` hàm `run(ds, cfg, preds=None)` so nhãn với dự đoán `checker` (`yolo26l`) để sinh 4 cảnh báo `WARN` (`POSSIBLE_MISSING`, `POSSIBLE_LOOSE_BOX`, `POSSIBLE_WRONG_CLASS`, `POSSIBLE_FALSE_POS`), dùng `ds.read_label_lines` giữ số dòng gốc, đọc 6 ngưỡng từ `cfg["second_opinion"]`.
- **Cách kiểm:** Kiểm tra thứ tự ưu tiên nhánh cùng lớp trước khác lớp, Pha B dùng `fp_conf` và không lọc `min_area_px`; chạy `python -m pytest tests/test_second_opinion.py -q` đạt `15 passed` và `python -m pytest -q` xanh toàn bộ.
- **Lưu ý / bẫy đã chặn:** Không dùng `ds.valid_boxes(stem)` vì làm mất số dòng gốc khi có dòng trắng/hỏng; nhánh mơ hồ khác lớp trong `[iou_missing, iou_match)` im lặng không sinh issue giả.

*(Thưởng +2, tùy chọn)* CVAT: lọc `width < 20`, ảnh/số box khớp, đối chiếu `BOX_TINY` (xem docs/CVAT_FILTER.md), 1 câu về `can_review`.
