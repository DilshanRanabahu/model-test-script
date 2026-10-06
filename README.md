# model-test-script

A single-file Python script that tests trained computer vision models on a test set and, when asked, benchmarks several models against each other.

For each model it reports accuracy, average inference time and per-image inference time. In benchmark mode it also draws a comparison chart to help you pick a model.

- **Architectures:** Ultralytics YOLO and RF-DETR. New ones can be added with one class.
- **Tasks:** object detection, instance segmentation, keypoint detection / pose estimation.
- **Model files:** `.pt`, `.pth` and `.onnx`.
- **Test data:** YOLO format or COCO format.

Everything is in [`model_benchmark.py`](model_benchmark.py).

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [The three main switches](#the-three-main-switches)
- [Test a single model](#test-a-single-model)
- [Benchmark several models](#benchmark-several-models)
- [Model files: .pt, .pth and .onnx](#model-files-pt-pth-and-onnx)
- [Describing a model: ModelSpec](#describing-a-model-modelspec)
- [Test data formats](#test-data-formats)
- [Tasks and metrics](#tasks-and-metrics)
- [Output](#output)
- [Run options](#run-options)
- [Using it from another script](#using-it-from-another-script)
- [Adding a new architecture](#adding-a-new-architecture)
- [Troubleshooting](#troubleshooting)
- [Known limits](#known-limits)
- [Supported task types](#supported-task-types)

## Requirements

Python 3 (developed and tested on Python 3.12), plus:

| Library | Minimum version | Needed for |
|---|---|---|
| `numpy` | 1.24.0 | always |
| `torch` | 2.0.0 | always |
| `supervision` | 0.26.0 | always |
| `opencv-python` | 4.8.0 | always |
| `pyyaml` | 6.0 | always |
| `ultralytics` | 8.3.0 | YOLO models |
| `rfdetr` | 1.2.0 | RF-DETR models |
| `matplotlib` | 3.7.0 | the benchmark chart |
| `onnxruntime` or `onnxruntime-gpu` | 1.16.0 | `.onnx` models |

Install everything:

```
pip install "numpy>=1.24.0" "torch>=2.0.0" "supervision>=0.26.0" "opencv-python>=4.8.0" "pyyaml>=6.0" "ultralytics>=8.3.0" "rfdetr>=1.2.0" "matplotlib>=3.7.0" "onnxruntime>=1.16.0"
```

For GPU support, install the `torch` build that matches your CUDA version: https://pytorch.org/get-started/locally/

You don't have to check this by hand. The script checks before it starts, and only for what the selected run needs. If something is missing or too old, it lists it and prints the exact `pip install` command:

```
Some required libraries are missing or too old:

+-------------+----------+---------------+
| Library     | Required | Status        |
+-------------+----------+---------------+
| rfdetr      | >= 1.2.0 | not installed |
| ultralytics | >= 8.3.0 | found 8.0.20  |
+-------------+----------+---------------+

Install them with:

  pip install "rfdetr>=1.2.0" "ultralytics>=8.3.0"
```

## Quick start

1. Open `model_benchmark.py` and find the block marked `USER CONFIGURATION - START`. It is the only part you edit.
2. Set the three main switches (step 1 of that block).
3. Put your model and test-set paths in the model list for your task (step 2).
4. Run:

```
python model_benchmark.py
```

## The three main switches

They sit together at the top of the configuration block:

```python
TASK = "detection"
BENCHMARK = True
SINGLE_MODEL = None
```

| Switch | Values | Meaning |
|---|---|---|
| `TASK` | `"detection"`, `"segmentation"`, `"keypoint"` (or `"pose"`) | What kind of models you are testing |
| `BENCHMARK` | `True` / `False` | `True` compares every model in the list and draws a chart. `False` tests one model. |
| `SINGLE_MODEL` | a model's `name`, or `None` | Which model to test when `BENCHMARK = False`. `None` means the first one in the list. |

There is one model list per task: `DETECTION_MODELS`, `SEGMENTATION_MODELS` and `KEYPOINT_MODELS`. The script uses the list that matches `TASK`, so switching task does not mean rewriting your models.

## Test a single model

Use this when you want the numbers for one model and no comparison.

```python
TASK = "detection"
BENCHMARK = False
SINGLE_MODEL = "My YOLO"

DETECTION_MODELS = [
    ModelSpec(
        name="My YOLO",
        architecture="yolo",
        weights=r"C:\models\yolo\best.pt",
        data_path=r"C:\datasets\my_data\data.yaml",
        data_format="yolo",
    ),
]
```

Run `python model_benchmark.py`. You get a results table for that model, plus `summary.json` and `per_image_times.csv`. No chart is drawn in this mode.

`SINGLE_MODEL` must match a model's `name` exactly. If it doesn't, the script stops and lists the valid names.

## Benchmark several models

Use this to compare models and decide between them.

```python
TASK = "detection"
BENCHMARK = True

DETECTION_MODELS = [
    ModelSpec(
        name="YOLO",
        architecture="yolo",
        weights=r"C:\models\yolo\best.pt",
        data_path=r"C:\datasets\my_data_yolo\data.yaml",
        data_format="yolo",
    ),
    ModelSpec(
        name="RF-DETR",
        architecture="rfdetr",
        weights=r"C:\models\rfdetr\checkpoint_best_total.pth",
        data_path=r"C:\datasets\my_data_coco\test\_annotations.coco.json",
        data_format="coco",
    ),
]
```

Run `python model_benchmark.py`. Every model in the list is tested one after another, then you get:

- a table with one column per model and a "Best" column,
- a "Most accurate" and "Fastest" summary,
- `benchmark.png`, which opens in your image viewer.

A benchmark needs at least two models and the chart supports up to eight. For a fair comparison, score all models on the same test images and the same device; the script warns you if they differ.

## Model files: .pt, .pth and .onnx

The script does not care about the extension for PyTorch weights. It passes the path to the library for that architecture. A file ending in `.onnx` is run with ONNX Runtime instead.

| Architecture | File | Detection | Segmentation | Keypoint / pose |
|---|---|---|---|---|
| YOLO | `.pt` | yes | yes | yes |
| YOLO | `.onnx` | yes | yes | yes |
| RF-DETR | `.pth` / `.pt` | yes | yes | yes |
| RF-DETR | `.onnx` | yes | no | no |

The file must match the architecture: a YOLO `.pt` goes with `architecture="yolo"`, an RF-DETR `.pth` with `architecture="rfdetr"`.

### YOLO `.pt`

```python
ModelSpec(
    name="YOLO",
    architecture="yolo",
    weights=r"C:\models\yolo\best.pt",
    data_path=r"C:\datasets\my_data\data.yaml",
    data_format="yolo",
)
```

Use detection weights for `TASK = "detection"`, `-seg` weights for segmentation and `-pose` weights for keypoints.

### RF-DETR `.pth`

```python
ModelSpec(
    name="RF-DETR",
    architecture="rfdetr",
    weights=r"C:\models\rfdetr\checkpoint_best_total.pth",
    data_path=r"C:\datasets\my_data_coco\test\_annotations.coco.json",
    data_format="coco",
)
```

The model size is read from the checkpoint. If it can't be worked out, name it yourself:

```python
options={"variant": "RFDETRNano"}
```

Valid names include `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, `RFDETRBase`, `RFDETRLarge` for detection, `RFDETRSegNano`, `RFDETRSegSmall`, `RFDETRSegMedium`, `RFDETRSegLarge` for segmentation and `RFDETRKeypointPreview` for keypoints.

### YOLO `.onnx`

Export first:

```
yolo export model=best.pt format=onnx
```

Then:

```python
ModelSpec(
    name="YOLO ONNX",
    architecture="yolo",
    weights=r"C:\models\yolo\best.onnx",
    data_path=r"C:\datasets\my_data\data.yaml",
    data_format="yolo",
)
```

Nothing extra is needed: Ultralytics stores the class names inside the ONNX file.

### RF-DETR `.onnx`

Export first:

```python
from rfdetr import RFDETRNano

model = RFDETRNano(pretrain_weights=r"C:\models\rfdetr\checkpoint_best_total.pth")
model.export(output_dir="onnx_out")
```

Then:

```python
ModelSpec(
    name="RF-DETR ONNX",
    architecture="rfdetr",
    weights=r"C:\models\rfdetr\model.onnx",
    data_path=r"C:\datasets\my_data\data.yaml",
    data_format="yolo",
    options={"class_names": ["cat", "dog"]},
)
```

`class_names` is required here. An RF-DETR ONNX file does not contain the class names, so list them in the model's class order, or as `{id: name}`. For a model pretrained on COCO, use `"class_names": "coco"`. Without names the script matches raw class ids, which can give wrong scores; it prints a note when that happens.

RF-DETR `.onnx` works for detection only. For segmentation or keypoints, use the `.pth` checkpoint.

### Comparing formats

To check that an export kept its accuracy, put the original and the exported model in the same list and benchmark them:

```python
DETECTION_MODELS = [
    ModelSpec(name="YOLO .pt",   architecture="yolo", weights=r"C:\models\best.pt",   data_path=DATA, data_format="yolo"),
    ModelSpec(name="YOLO .onnx", architecture="yolo", weights=r"C:\models\best.onnx", data_path=DATA, data_format="yolo"),
]
```

### ONNX and the GPU

ONNX models use the GPU only if `onnxruntime-gpu` is installed and matches your CUDA and cuDNN versions. If it can't use the GPU, the model runs on the CPU. The script does not crash: the Device row shows `cpu (onnx)`, a note explains why, and it warns that timings are not comparable with models that ran on the GPU. Accuracy is not affected.

## Describing a model: ModelSpec

| Field | Meaning |
|---|---|
| `name` | Label used in the table and chart. Also what `SINGLE_MODEL` refers to. Must be unique. |
| `architecture` | `"yolo"` or `"rfdetr"` |
| `weights` | Path to the `.pt`, `.pth` or `.onnx` file |
| `data_path` | Path to the test data for this model (see below) |
| `data_format` | `"yolo"` or `"coco"` |
| `split` | Which split of a YOLO dataset to use. Default `"test"`. |
| `options` | Optional extras, listed below |

Options:

| Option | Applies to | Meaning |
|---|---|---|
| `images_dir`, `labels_dir` | any | Read images / YOLO labels from these folders instead of the ones in the dataset file |
| `class_names` | any | The model's class names, if the weights don't carry them |
| `allow_download` | any | `True` lets the library download pretrained weights by name, for example `weights="yolo11n.pt"` |
| `imgsz` | YOLO | Inference image size |
| `variant` | RF-DETR | Model class name, for example `"RFDETRNano"` |
| `resolution` | RF-DETR | Inference resolution |
| `optimize` | RF-DETR | `True` calls `optimize_for_inference()` |
| `trust_checkpoint` | RF-DETR | `True` allows full-pickle loading of a checkpoint you trust |
| `kpt_sigmas` | keypoint tasks | Per-keypoint tolerances for scoring |

## Test data formats

Each model reads its own test data, in the format you give it.

**YOLO format.** Point `data_path` at the `data.yaml` (or its folder). The script reads the split named in `split`, and finds labels by replacing `images` with `labels` in the path.

```
my_data/
  data.yaml
  images/test/*.jpg
  labels/test/*.txt
```

**COCO format.** Point `data_path` at the annotations `.json`, or at the folder holding `_annotations.coco.json`. Images are expected in the same folder as the file unless you set `images_dir`.

```
my_data_coco/test/
  _annotations.coco.json
  *.jpg
```

Two things make this easier:

- **One copy is enough.** The script feeds images to each model itself, so both models can point at the same dataset in the same format.
- **Classes are matched by name.** Different exports often number classes differently, so the script matches the model's class names to the dataset's class names, not the ids.

Labels must suit the task: boxes for detection, polygons or masks for segmentation, keypoints for keypoint detection.

## Tasks and metrics

| Task | Accuracy metrics reported |
|---|---|
| `detection` | mAP50-95, mAP50, precision, recall, F1 |
| `segmentation` | mask mAP50-95, mask mAP50, mask precision, recall, F1, plus box mAP50-95 |
| `keypoint` / `pose` | keypoint mAP50-95, keypoint mAP50, precision, recall, F1 |

- mAP is measured over the whole confidence range.
- Precision, recall and F1 are measured at `CONF_THRESHOLD`, averaged over classes.
- Keypoints are scored with OKS (object keypoint similarity), the keypoint counterpart of box overlap, as COCO does.
- Every architecture goes through the same scorer, so the numbers are comparable between models. They can differ slightly from each library's own validation command.

Speed, for every task:

| Metric | Meaning |
|---|---|
| Avg / median / p95 / min / max inference (ms) | Time per image |
| Throughput (FPS) | Images per second, from the average |
| Avg forward only (ms) | The network alone, when the library reports it (YOLO) |

How timing is done: one image at a time (batch size 1), image already loaded in memory, after `WARMUP_RUNS` untimed predictions. The time covers the library's whole `predict` call, so it includes resizing and decoding the output, but not reading the image from disk.

## Output

In the terminal you get a progress bar per model and a results table:

```
+-----------------------+---------+--------------+--------------+
| Metric                | YOLO11n | RF-DETR Nano | Best         |
+-----------------------+---------+--------------+--------------+
| Accuracy              |         |              |              |
| mAP50-95              |   0.614 |        0.734 | RF-DETR Nano |
| mAP50                 |   0.856 |        0.901 | RF-DETR Nano |
| ...                   |         |              |              |
+-----------------------+---------+--------------+--------------+
| Speed                 |         |              |              |
| Avg inference (ms)    |    78.0 |        129.4 | YOLO11n      |
| ...                   |         |              |              |
+-----------------------+---------+--------------+--------------+
  Most accurate  RF-DETR Nano  mAP50-95 0.734
  Fastest        YOLO11n  78.0 ms per image (12.8 FPS)
```

(The numbers above are from pretrained nano models on a 4-image sample. They show the layout, not what to expect from your models.)

Files are saved in `results/<date_time>/`:

| File | Contents |
|---|---|
| `summary.json` | All metrics per model and the settings used |
| `per_image_times.csv` | Inference time of every image, per model |
| `benchmark.png` | The comparison chart (benchmark mode only) |

The chart has five parts: accuracy bars, inference time bars (average, median, p95), the per-image time distribution, a speed-versus-accuracy plot, and a summary table with the better model per metric in bold.

In an interactive terminal the output is coloured. If you redirect it to a file, colours are switched off automatically.

## Run options

Step 3 of the configuration block. The defaults are fine for most runs.

| Option | Default | Meaning |
|---|---|---|
| `DEVICE` | `"auto"` | `"auto"` uses the GPU if available, otherwise the CPU. Or `"cpu"`, `"cuda"`, `"cuda:0"`. |
| `CONF_THRESHOLD` | `0.25` | Confidence used for precision, recall, F1 and the timed pass |
| `NMS_IOU` | `0.7` | NMS overlap threshold for YOLO. RF-DETR does not use NMS. |
| `WARMUP_RUNS` | `5` | Untimed predictions before timing starts |
| `MAX_IMAGES` | `None` | Limit the number of test images, for example `20` for a quick trial |
| `OUTPUT_DIR` | `"results"` | Where results are saved |
| `SHOW_PLOT` | `True` | Open `benchmark.png` after saving it |

Memory: each model is released from RAM and GPU memory as soon as it has been tested, before the next one loads. If a model runs out of GPU memory, the script reruns it on the CPU and says so in the results.

## Using it from another script

Every setting can also be passed to the constructor:

```python
from model_benchmark import Evaluator, ModelSpec

models = [
    ModelSpec(name="YOLO", architecture="yolo", weights=r"C:\models\best.pt",
              data_path=r"C:\datasets\my_data\data.yaml", data_format="yolo"),
    ModelSpec(name="RF-DETR", architecture="rfdetr", weights=r"C:\models\checkpoint.pth",
              data_path=r"C:\datasets\my_data\data.yaml", data_format="yolo"),
]

results = Evaluator(task="detection", benchmark=True, models=models).run()

for result in results:
    print(result.model_name, result.metrics["map50_95"], result.timing["avg_ms"])
```

Anything you leave out takes its value from the configuration block. `run()` returns one result object per model with `metrics`, `timing`, `per_image_ms` and `notes`.

## Adding a new architecture

Three steps, all in `model_benchmark.py`. Nothing else needs to change; the benchmark table and chart pick the new model up automatically.

**1. Write an adapter** next to `YoloAdapter` and `RFDETRAdapter`:

```python
@register_architecture("myarch")
class MyArchAdapter(ModelAdapter):
    supported_tasks = {"detection"}

    def load(self) -> None:
        import mylib
        self.model = mylib.load(self._weights_path(), device=self.device)

    @property
    def class_names(self) -> dict[int, str]:
        return self._named_classes(self.model.names)

    def prepare(self, image_bgr):        # optional, not timed
        return image_bgr

    def predict(self, prepared, conf: float):   # timed
        return self.model(prepared, conf=conf)

    def to_output(self, raw):            # not timed
        import supervision as sv
        return sv.Detections(xyxy=raw.boxes, confidence=raw.scores, class_id=raw.labels)
```

`mylib` and the `raw.*` fields stand for your library's real calls. `to_output` returns `sv.Detections` for detection, the same with `mask` for segmentation, or a `KeypointSet` for keypoints.

**2. Add its library** to the `REQUIRED` list:

```python
Requirement(("mylib",), "1.0.0", "myarch"),
```

**3. Use it** in a model list with `architecture="myarch"`.

New data formats and tasks are added the same way, with `@register_data_format("name")` and `@register_task("name")`.

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `Some required libraries are missing or too old` | Run the `pip install` command printed below the table |
| `weights file not found` | The `weights` path is wrong. To download pretrained weights by name, add `options={"allow_download": True}`. |
| `YOLO data.yaml not found` / `COCO annotations not found` | The `data_path` is wrong |
| `data.yaml has no 'test' folder entry` | Set `split="val"` on the `ModelSpec`, or give `images_dir` and `labels_dir` in `options` |
| `SINGLE_MODEL '...' is not in MODELS` | The name doesn't match any model for the current task. The message lists the valid names. |
| `BENCHMARK is True but fewer than two models are configured` | Add a second model, or set `BENCHMARK = False` |
| `the model returned boxes but no masks` | `TASK = "segmentation"` needs segmentation weights |
| `the model returned no keypoints` | `TASK = "keypoint"` needs pose / keypoint weights |
| `RF-DETR .onnx models are supported for task 'detection' only` | Use the `.pth` checkpoint for that task |
| `Task 'classification' is not implemented yet` | Classification is a reserved name, not built yet |
| Note: `No class names in common between model and dataset` | Give the model's names with `options={"class_names": [...]}` |
| Note: `ONNX Runtime could not use the GPU` | Install an `onnxruntime-gpu` that matches your CUDA version, or accept CPU timings |
| Warning: `the models ran on different devices` | Timings are not comparable between those models; accuracy still is |

## Known limits

- **Classification** is not implemented.
- **RF-DETR `.onnx`** supports detection only.
- **Keypoint precision** can read low for a model that also finds people who have no labelled keypoints in the test set: those people are dropped from the ground truth, so correct detections of them count as wrong. Keypoint mAP is less affected.
- **Large segmentation test sets** can use a lot of memory, because all ground-truth masks are loaded before testing starts.
- **Batch size** is always 1.

## Supported task types

A summary of what the script can and cannot test today.

### Supported

| Task type | `TASK` value | What the model outputs | YOLO | RF-DETR |
|---|---|---|---|---|
| Object detection | `"detection"` | A box and a class for each object | `.pt`, `.onnx` | `.pth` / `.pt`, `.onnx` |
| Instance segmentation | `"segmentation"` | A box, a class and a pixel mask for each object | `.pt`, `.onnx` | `.pth` / `.pt` |
| Keypoint detection / pose estimation | `"keypoint"` or `"pose"` | A set of keypoints (for example body joints) for each object | `.pt`, `.onnx` | `.pth` / `.pt` |

Notes on each:

- **Object detection** uses ordinary upright boxes (`x1, y1, x2, y2`). Any number of classes.
- **Instance segmentation** gives each object its own mask, so two touching objects of the same class stay separate. Masks are scored by mask overlap, and box accuracy is reported alongside.
- **Keypoint detection and pose estimation** are the same task here; the two `TASK` names behave identically. Any number of keypoints per object is accepted. Models with 17 keypoints are scored with the standard COCO person tolerances; other keypoint counts use a uniform tolerance, which you can change with the `kpt_sigmas` option.

### Not supported

| Task type | Status |
|---|---|
| Image classification | Reserved name (`"classification"`); the script stops with a "not implemented yet" message |
| Semantic segmentation (one class label per pixel, no separate objects) | Not supported |
| Panoptic segmentation | Not supported |
| Oriented (rotated) box detection | Not supported |
| Object tracking in video | Not supported; the script scores still images only |

To add a task, subclass `TaskEvaluator` and register it with `@register_task("name")`, as described in [Adding a new architecture](#adding-a-new-architecture).
