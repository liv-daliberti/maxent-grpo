"""Render fixed figure layouts from unchanged frozen JSON; record text geometry."""
from pathlib import Path
import importlib.util
import json
import shutil
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
PHASE = sys.argv[1] if len(sys.argv) > 1 else "after"
STEMS = {4: ("plot_paper_experiment1_composite.py", "experiment1_retention_comparator_matrix"),
         12: ("plot_paper_replay_mechanism_telemetry.py", "replay_mechanism_telemetry_qwen05b")}
geometry = {}
for number, (source, stem) in STEMS.items():
    spec = importlib.util.spec_from_file_location(f"overlap_figure_{number}", ROOT / "ops/exp_scaling" / source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    payload_path = ROOT / "paper/figures" / f"{stem}.json"
    original_bytes = payload_path.read_bytes()
    payload = json.loads(original_bytes)
    original_save = module.style.save
    def capture(figure, output, **kwargs):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        if number == 4:
            axis_gaps = []
            for axis in figure.axes:
                labels = axis.get_xticklabels()
                bboxes = [label.get_window_extent(renderer) for label in labels]
                axis_gaps.append([right.x0 - left.x1 for left, right in zip(bboxes, bboxes[1:])])
            geometry[str(number)] = {"adjacent_tick_gaps_pixels": axis_gaps,
                                     "minimum_tick_gap_points": min(min(row) for row in axis_gaps) * 72 / figure.dpi}
        else:
            axis = figure.axes[1]
            text = next(text for text in axis.texts if text.get_text() == "capacity 16")
            annotation = text.get_window_extent(renderer)
            title = axis.title.get_window_extent(renderer)
            geometry[str(number)] = {"title_bbox": list(title.bounds), "capacity_bbox": list(annotation.bounds),
                                     "title_capacity_overlap": bool(annotation.overlaps(title)),
                                     "title_capacity_vertical_gap_points": (title.y0 - annotation.y1) * 72 / figure.dpi,
                                     "capacity_fully_inside_axes": bool(axis.bbox.contains(annotation.x0, annotation.y0) and axis.bbox.contains(annotation.x1, annotation.y1))}
        original_save(figure, output, **kwargs)
    module.style.save = capture
    output = ROOT / "paper/figures" / stem if PHASE == "after" else AUDIT / "baseline_recreated" / stem
    module.render(payload, output)
    module.style.save = original_save
    assert payload_path.read_bytes() == original_bytes
    geometry[str(number)]["data_sha256"] = hashlib.sha256(original_bytes).hexdigest()
    if PHASE == "after":
        shutil.copy2(output.with_suffix(".pdf"), ROOT / "paper/mathai2026/figures" / f"{stem}.pdf")
(AUDIT / f"geometry_{PHASE}.json").write_text(json.dumps(geometry, indent=2) + "\n")
print(json.dumps(geometry, indent=2))
