"""BƯỚC 4 — Cửa chặn: quyết định Release / Hold / Rework.   ★

Test:  pytest tests/test_gate.py -q

Cây quyết định Day 07 (slide Day 09: "Lỗi bắt buộc chặn pipeline; lỗi cảnh báo chuyển người xem"):

  còn ≥ 1 lỗi bắt buộc (severity ERROR)                          -> "Rework"
  không còn lỗi bắt buộc, nhưng tỉ lệ ẢNH còn cảnh báo
      >= cfg["gate"]["warn_ratio_hold"]                          -> "Hold"
  còn lại                                                        -> "Release"

Tỉ lệ cảnh báo = (số ẢNH khác nhau có ≥ 1 cảnh báo) / (tổng số ảnh).  Chú ý:
- chỉ đếm issue có `image` nằm trong `image_stems` (bỏ "<dataset>", "data.yaml");
- bỏ qua ảnh nằm trong `reviewed` (người đã xem, cảnh báo của ảnh đó coi như đã xử lý) — nhưng LỖI BẮT BUỘC thì không bao giờ được bỏ qua;
- mẫu số là số ảnh, không phải số issue.

Trả về dict: decision, n_errors, n_warn_images, n_images, warn_ratio (làm tròn 4 chữ số), reasons (list[str], giải thích bằng tiếng Việt, có số liệu).
"""
from ..common import ERROR, WARN  # noqa: F401


def decide(issues, image_stems, cfg, reviewed=frozenset()):
    """Quyết định Release / Hold / Rework dựa trên lỗi bắt buộc và tỉ lệ ảnh còn cảnh báo."""
    hold_ratio = cfg["gate"]["warn_ratio_hold"]
    stem_set = set(image_stems)
    n_images = len(image_stems)

    n_errors = sum(1 for i in issues if i.severity == ERROR)
    warn_images = {
        i.image
        for i in issues
        if i.severity == WARN and i.image in stem_set and i.image not in reviewed
    }
    n_warn_images = len(warn_images)
    warn_ratio = round(n_warn_images / n_images, 4) if n_images else 0.0

    if n_errors > 0:
        decision = "Rework"
        reasons = [f"còn {n_errors} lỗi bắt buộc (ERROR)"]
    elif warn_ratio >= hold_ratio:
        decision = "Hold"
        reasons = [f"{n_warn_images}/{n_images} ảnh ({warn_ratio:.2%}) còn cảnh báo >= ngưỡng {hold_ratio:.0%}"]
    else:
        decision = "Release"
        reasons = [f"không còn lỗi bắt buộc; {n_warn_images}/{n_images} ảnh ({warn_ratio:.2%}) còn cảnh báo < {hold_ratio:.0%}"]

    return {
        "decision": decision,
        "n_errors": n_errors,
        "n_warn_images": n_warn_images,
        "n_images": n_images,
        "warn_ratio": warn_ratio,
        "reasons": reasons,
    }
