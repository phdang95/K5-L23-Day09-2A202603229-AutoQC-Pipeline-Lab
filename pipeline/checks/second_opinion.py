"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401


def _read_valid_labels(ds, stem: str) -> list[tuple[int, int, tuple[float, float, float, float]]] | None:
    """Đọc nhãn hợp lệ kèm số dòng gốc (từ 1); trả None nếu không có file nhãn."""
    lines = ds.read_label_lines(stem)
    if lines is None:
        return None
    n_classes = len(ds.classes)
    valid = []
    for line_no, raw in lines:
        try:
            cls_id, cx, cy, w, h = parse_yolo_line(raw)
        except ValueError:
            continue
        if 0 <= cls_id < n_classes and w > 0 and h > 0:
            valid.append((line_no, cls_id, (cx, cy, w, h)))
    return valid


def run(ds, cfg, preds=None):
    """So nhãn với dự đoán model lớn hơn để cảnh báo thiếu/nhầm lớp/box lỏng/dư."""
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    so_cfg = cfg["second_opinion"]
    conf_thr = so_cfg["conf"]
    min_area_px = so_cfg["min_area_px"]
    fp_conf = so_cfg["fp_conf"]
    iou_missing = so_cfg["iou_missing"]
    iou_match = so_cfg["iou_match"]
    iou_loose = so_cfg["iou_loose"]

    issues = []
    for img_path in ds.image_files():
        stem = img_path.stem
        if stem not in preds:
            continue
        labels = _read_valid_labels(ds, stem)
        if labels is None:
            continue
        img_size = ds.image_size(stem)
        if img_size is None:
            continue
        img_w, img_h = img_size
        raw_preds = preds[stem]

        # Pha A: kiểm từng dự đoán đạt ngưỡng conf và min_area_px
        for p in raw_preds:
            p_cls, p_conf, pcx, pcy, pw, ph = p
            area_px = (pw * img_w) * (ph * img_h)
            if p_conf < conf_thr or area_px < min_area_px:
                continue
            p_box = (pcx, pcy, pw, ph)
            cands = []
            for _, l_cls, l_box in labels:
                iou = iou_xywh(p_box, l_box)
                if iou >= iou_missing:
                    cands.append((l_cls, iou))

            if not cands:
                issues.append(
                    Issue("second_opinion", WARN, "POSSIBLE_MISSING", stem, f"model thấy lớp {p_cls} (conf={p_conf:.2f}) nhưng thiếu nhãn", extra={"conf": p_conf})
                )
                continue

            same_ious = [iou for l_cls, iou in cands if l_cls == p_cls]
            if same_ious:
                best_same = max(same_ious)
                if best_same < iou_loose:
                    issues.append(
                        Issue("second_opinion", WARN, "POSSIBLE_LOOSE_BOX", stem, f"box lớp {p_cls} lỏng (IoU={best_same:.2f})", extra={"conf": p_conf, "iou": best_same})
                    )
            else:
                best_diff = max(iou for _, iou in cands)
                if best_diff >= iou_match:
                    issues.append(
                        Issue("second_opinion", WARN, "POSSIBLE_WRONG_CLASS", stem, f"nghi nhầm lớp thành {p_cls} (IoU={best_diff:.2f})", extra={"conf": p_conf, "iou": best_diff})
                    )

        # Pha B: kiểm nhãn thừa (dùng fp_conf, KHÔNG lọc min_area_px)
        fp_preds = [(p[2], p[3], p[4], p[5]) for p in raw_preds if p[1] >= fp_conf]
        for line_no, l_cls, l_box in labels:
            if not any(iou_xywh(l_box, p_box) >= iou_missing for p_box in fp_preds):
                issues.append(
                    Issue("second_opinion", WARN, "POSSIBLE_FALSE_POS", stem, f"nhãn lớp {l_cls} không khớp dự đoán nào", line=line_no)
                )

    return issues
