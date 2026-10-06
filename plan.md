# Plan: single-file model evaluation and benchmark script (YOLO vs RF-DETR)

## Context

You need one script that scores trained models against a test set and, when asked, benchmarks two architectures (Ultralytics YOLO and RF-DETR) side by side with a chart that supports a pick-one decision. It must stay in a single file, be easy to extend with new architectures and tasks, and take its settings (task, benchmark flag, model paths, data paths, data formats) from variables or a constructor.

Starting point: `C:\Users\DilshanR\Desktop\Thullex\test_script` is empty. Installed already: Python 3.12, ultralytics 8.4.138, rfdetr 1.9.3, supervision 0.30.0, torch 2.7.1+cu118, matplotlib 3.11.1. GPU is an MX230 with 2 GB, so the script needs a CPU fallback.

Your answers: object detection must work on day one; accuracy = mAP50-95 / mAP50 plus precision, recall, F1; timing = batch 1, after warm-up, excluding disk loading; data format not settled, so both YOLO and COCO are supported. The sketch image did not arrive, so this plan is built from the written requirements only.

## File to create

`model_benchmark.py` (the only file). Outputs go to a `results/` folder next to it.

## Layout of the file, top to bottom

1. **User settings block** (plain variables, the only part most users touch)
   - `TASK = "detection"`
   - `BENCHMARK = True`
   - `MODELS = [ModelSpec(name="YOLO", architecture="yolo", weights=r"...", data_path=r"...", data_format="yolo"), ModelSpec(name="RF-DETR", architecture="rfdetr", weights=r"...", data_path=r"...", data_format="coco")]`
   - `SINGLE_MODEL = "YOLO"` (which one runs when `BENCHMARK = False`)
   - `DEVICE = "auto"`, `IMAGE_SIZE`, `CONF_THRESHOLD = 0.25`, `IOU_THRESHOLD`, `WARMUP_RUNS = 5`, `OUTPUT_DIR = "results"`
   - The same values can be passed to `Evaluator(task=..., benchmark=..., models=[...], ...)` when the file is imported instead of run.

2. **Dependency check** (`check_dependencies()`), run before any heavy import
   - Uses `importlib.metadata` to compare installed versions against a `REQUIRED` table (ultralytics, rfdetr, supervision, torch, matplotlib, numpy, pillow).
   - Checks only what the chosen architectures need, so a YOLO-only run does not demand rfdetr.
   - On a miss, prints each package, required version, found version, and the exact `pip install "pkg>=x.y"` line, then exits cleanly.
   - Heavy libraries are imported lazily inside the adapters for this reason.

3. **Three small registries** (the extension points)
   - `@register_architecture("yolo")` → `ModelAdapter` subclasses
   - `@register_data_format("yolo")` / `("coco")` → dataset loader functions
   - `@register_task("detection")` → `TaskEvaluator` subclasses
   - Adding an architecture later = one new class with `load()` and `predict(image)`; nothing else changes.

4. **Model adapters** (`ModelAdapter` base: `load()`, `predict(image, conf) -> sv.Detections`, `class_names`, `supported_tasks`)
   - `YoloAdapter`: `ultralytics.YOLO(weights)`, `model.predict(...)`, converted with `sv.Detections.from_ultralytics`. Also keeps Ultralytics' own forward-only time from `result.speed` as extra information.
   - `RFDETRAdapter`: `rfdetr.from_checkpoint(weights)`, with an optional `variant="RFDETRNano"` etc. in `ModelSpec.options` as a fallback; `model.predict(image, threshold=...)` already returns `sv.Detections`.
   - An adapter asked for a task it does not list fails with a plain message naming the architecture and task.

5. **Dataset loaders** → one common `sv.DetectionDataset`
   - YOLO: `sv.DetectionDataset.from_yolo(images, labels, data.yaml)`; COCO: `sv.DetectionDataset.from_coco(images, annotations.json)`.
   - Each model reads ground truth from its own path and format, as required. Because the script feeds images to `predict` itself, both models may also point at the same single copy, so no converter is needed.
   - Class IDs are matched **by class name** between model and dataset, because COCO exports often shift IDs by one; unmatched names are reported, not silently dropped.
   - Warns if the two models' test sets differ in image count.

6. **Task evaluators** (`TaskEvaluator` base: `evaluate(adapter, dataset) -> EvalResult`)
   - `DetectionEvaluator` (implemented): a timed pass at `CONF_THRESHOLD` (the deployment setting) that feeds `Precision`, `Recall`, `F1Score`, then an untimed pass at a very low confidence for mAP (`sv.metrics.MeanAveragePrecision`). Both models go through this identical scorer, so the numbers are comparable.
   - `SegmentationEvaluator` (implemented, added after the first version): subclasses `DetectionEvaluator`, reads mask annotations, scores IoU on masks (mask mAP, precision, recall, F1) and also reports box mAP50-95. Masks are stored run-length encoded (`sv.CompactMask`) to keep memory down.
   - `pose`, `keypoint`, `classification`: registered as named stubs that raise a clear "not implemented yet, add it here" message. Pose reuses the same loop when added later (keypoint targets); classification needs top-1/top-5 and a folder-per-class loader.

7. **Timing** (shared by every task)
   - Image is loaded into memory first; the clock wraps only the `predict` call, batch 1, with `torch.cuda.synchronize()` on GPU, after `WARMUP_RUNS` discarded runs.
   - This wall-clock figure includes each library's resize and box decoding, which is the only way to time both libraries identically; YOLO's forward-only figure is reported next to it as a reference.
   - Reported: average, median, p95, min, max, FPS, and the full per-image list.

8. **Reporting**
   - Console: a table per model (mAP50-95, mAP50, precision, recall, F1, avg / median / p95 ms, FPS, device, image count).
   - `results/<run>/summary.json`, and `per_image_times.csv` (image name, ms per model).

9. **Benchmark mode** (`BENCHMARK = True`)
   - Runs every entry in `MODELS`, then draws one Matplotlib figure saved as `benchmark.png` (and shown if a display is available):
     - grouped bars for the accuracy metrics
     - average inference time bars with p95 markers
     - per-image inference time distribution (box plot)
     - speed vs accuracy scatter (avg ms vs mAP50-95), the decision view
     - a small summary table marking the better model per metric
   - `BENCHMARK = False` runs `SINGLE_MODEL` only and prints and saves its report with no comparison figure.
   - The `dataviz` skill is loaded before writing the chart code.

10. **`Evaluator` class and `main()`**: validates paths and settings with readable errors, resolves the device (falls back to CPU if CUDA is missing or runs out of memory on the 2 GB card), runs, reports.

## Verification

1. `python model_benchmark.py` with a deliberately wrong package version in `REQUIRED` → confirm the install message is correct, then restore.
2. Wrong path / unsupported task (`TASK = "classification"`) → confirm the messages are clear and no traceback wall.
3. Real run, `BENCHMARK = False`, on your YOLO weights and test set → check mAP against `yolo val` on the same data as a sanity reference.
4. Real run, `BENCHMARK = True`, with both models → confirm console table, `summary.json`, `per_image_times.csv`, and `benchmark.png`.
5. If your weights and test set are not ready yet, I will smoke-test with the pretrained `yolo11n.pt` and RF-DETR Nano on the small `coco8` sample (needs a one-time download) and say plainly that this is what was tested.

## Needed from you before or during the build

- Paths to the two weight files and the test set(s), and which RF-DETR size was trained (Nano / Small / Medium / Base / Large).
- The sketch image, if it shows anything that differs from the above.
