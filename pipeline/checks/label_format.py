"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


def run(ds, cfg):
    """Kiểm tra cú pháp và hình học từng dòng nhãn YOLO trong dataset `ds`."""
    lf_cfg = cfg["label_format"]
    overflow_tol = lf_cfg["overflow_tol"]
    min_box_px = lf_cfg["min_box_px"]
    max_box_frac = lf_cfg["max_box_frac"]
    dup_box_iou = lf_cfg["dup_box_iou"]
    num_classes = len(ds.classes)

    issues = []
    for label_file in ds.label_files():
        stem = label_file.stem
        lines = ds.read_label_lines(stem)
        if lines is None:
            continue

        img_size = ds.image_size(stem)
        seen_boxes = []

        for line_no, raw in lines:
            try:
                cls_id, cx, cy, w, h = parse_yolo_line(raw)
            except ValueError as exc:
                issues.append(Issue("label_format", ERROR, "BAD_FORMAT", stem, f"dòng sai định dạng: {exc}", line=line_no))
                continue

            if not (0 <= cls_id < num_classes):
                issues.append(Issue("label_format", ERROR, "BAD_CLASS_ID", stem, f"id lớp ngoài phạm vi: {cls_id}", line=line_no))
                continue

            if w <= 0 or h <= 0:
                issues.append(Issue("label_format", ERROR, "BOX_ZERO", stem, f"box rỗng: w={w}, h={h}", line=line_no))
                continue

            overflow = max(
                0.0 - (cx - w / 2),
                (cx + w / 2) - 1.0,
                0.0 - (cy - h / 2),
                (cy + h / 2) - 1.0,
                0.0,
            )
            if overflow > overflow_tol:
                issues.append(
                    Issue(
                        "label_format",
                        ERROR,
                        "BOX_OUT_OF_IMAGE",
                        stem,
                        f"box tràn ngoài ảnh ({overflow:.4f})",
                        line=line_no,
                        extra={"overflow": overflow},
                    )
                )

            cur_box = (cx, cy, w, h)
            if any(prev_cls == cls_id and iou_xywh(prev_box, cur_box) > dup_box_iou for prev_cls, *prev_box in seen_boxes):
                issues.append(Issue("label_format", ERROR, "DUP_BOX", stem, "box trùng lặp với dòng trước", line=line_no))
            seen_boxes.append((cls_id, cx, cy, w, h))

            if img_size is not None:
                img_w, img_h = img_size
                short_side = min(w * img_w, h * img_h)
                if short_side < min_box_px:
                    issues.append(Issue("label_format", WARN, "BOX_TINY", stem, f"cạnh ngắn {short_side:.1f}px < {min_box_px}px", line=line_no))

            if w * h > max_box_frac:
                issues.append(Issue("label_format", WARN, "BOX_HUGE", stem, f"diện tích {w * h:.2f} > {max_box_frac}", line=line_no))

    return issues
