"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
REMOVE_CODES = {"CORRUPT_IMAGE", "BAD_SIZE", "LEAK_TEST"}


def _is_remove_issue(code: str) -> bool:
    return code in REMOVE_CODES or code.startswith("DUP_IMAGE_")


def _clip_yolo_box(raw: str) -> str | None:
    """Cắt box YOLO về [0, 1]; trả None nếu box nằm hẳn ngoài ảnh."""
    cls_id, cx, cy, w, h = parse_yolo_line(raw)
    x1 = max(0.0, cx - w / 2)
    y1 = max(0.0, cy - h / 2)
    x2 = min(1.0, cx + w / 2)
    y2 = min(1.0, cy + h / 2)
    if x2 <= x1 or y2 <= y1:
        return None
    ncx, ncy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = x2 - x1, y2 - y1
    return f"{cls_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}"


def fix(ds, issues, cfg):
    """Sửa tự động các lỗi ERROR an toàn trên bản sao `ds` và trả về log thay đổi."""
    err_issues = [i for i in issues if i.severity == ERROR]
    fixes = []

    removed_stems = set()
    for iss in err_issues:
        if _is_remove_issue(iss.code) and iss.image not in removed_stems:
            removed_stems.add(iss.image)
            img_p = ds.image_path(iss.image)
            lbl_p = ds.label_path(iss.image)
            if img_p is not None and img_p.exists():
                img_p.unlink()
            if lbl_p.exists():
                lbl_p.unlink()
            fixes.append({
                "image": iss.image,
                "line": "",
                "action": "remove_image",
                "before": iss.code,
                "after": "đã xoá ảnh và nhãn",
            })

    if any(iss.code == "BAD_CLASS_NAME" for iss in err_issues):
        data = {}
        if ds.yaml_path.exists():
            data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
        before_names = str(data.get("names", ""))
        data["names"] = list(cfg["classes"])
        ds.yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        fixes.append({
            "image": "data.yaml",
            "line": "",
            "action": "rename_classes",
            "before": before_names,
            "after": str(data["names"]),
        })

    for iss in err_issues:
        if iss.code == "MISSING_LABEL_FILE" and iss.image not in removed_stems:
            fixes.append({
                "image": iss.image,
                "line": "",
                "action": "needs_label",
                "before": "(không có file)",
                "after": "chuyển người gán",
            })

    drop_by_stem = defaultdict(set)
    clip_by_stem = defaultdict(set)
    for iss in err_issues:
        if iss.image in removed_stems or iss.line is None:
            continue
        if iss.code in DROP_CODES:
            drop_by_stem[iss.image].add(iss.line)
        elif iss.code == "BOX_OUT_OF_IMAGE":
            clip_by_stem[iss.image].add(iss.line)

    for stem in sorted(set(drop_by_stem) | set(clip_by_stem)):
        lbl_p = ds.label_path(stem)
        if not lbl_p.exists():
            continue
        raw_lines = lbl_p.read_text(encoding="utf-8", errors="replace").splitlines()
        drops = drop_by_stem[stem]
        clips = clip_by_stem[stem]
        new_lines = []

        for line_no, raw in enumerate(raw_lines, 1):
            if line_no in drops:
                fixes.append({
                    "image": stem,
                    "line": line_no,
                    "action": "drop_line",
                    "before": raw,
                    "after": "",
                })
            elif line_no in clips:
                clipped = _clip_yolo_box(raw)
                if clipped is None:
                    fixes.append({
                        "image": stem,
                        "line": line_no,
                        "action": "drop_line",
                        "before": raw,
                        "after": "",
                    })
                else:
                    fixes.append({
                        "image": stem,
                        "line": line_no,
                        "action": "clip_box",
                        "before": raw,
                        "after": clipped,
                    })
                    new_lines.append(clipped)
            else:
                new_lines.append(raw)

        lbl_p.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

    return fixes
