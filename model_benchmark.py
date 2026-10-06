"""Evaluate computer-vision models against a test set and benchmark them.

Run it directly after editing the USER SETTINGS block below:

    python model_benchmark.py

or import it and pass the same settings to the constructor:

    from model_benchmark import Evaluator, ModelSpec
    Evaluator(task="detection", benchmark=True, models=[ModelSpec(...), ModelSpec(...)]).run()

Extending the script
--------------------
* New architecture -> subclass ``ModelAdapter`` and decorate it with ``@register_architecture("name")``.
* New data format  -> write a loader and decorate it with ``@register_data_format("name")``.
* New task         -> subclass ``TaskEvaluator`` and decorate it with ``@register_task("name")``.
Nothing else in the file needs to change.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

# Heavy libraries (torch, ultralytics, rfdetr, supervision, matplotlib, cv2, numpy) are imported
# lazily inside the code that needs them, so check_dependencies() can report a missing or outdated
# package with a readable message instead of an ImportError.


@dataclass
class ModelSpec:
    """One model to evaluate.

    name:         label used in the report and the chart.
    architecture: a registered architecture ("yolo", "rfdetr").
    weights:      path to the trained weights file.
    data_path:    test data for this model. YOLO format: the data.yaml (or its folder).
                  COCO format: the annotations .json (or the folder holding _annotations.coco.json).
    data_format:  a registered data format ("yolo", "coco").
    split:        which split of the dataset to use ("test", "val", "valid", "train").
    options:      architecture / loader specific extras, all optional:
                    any         images_dir, labels_dir  override where images / YOLO labels are read from
                                class_names             list of class names if the weights do not carry them
                                allow_download          True lets the library fetch pretrained weights by name
                    yolo        imgsz                   inference image size
                    rfdetr      variant                 class name, e.g. "RFDETRNano", if it cannot be inferred
                                resolution              inference resolution
                                optimize                True to call optimize_for_inference()
                                trust_checkpoint        True to allow full-pickle loading of a trusted checkpoint
    """

    name: str
    architecture: str
    weights: str
    data_path: str
    data_format: str
    split: str = "test"
    options: dict[str, Any] = field(default_factory=dict)


# ======================================================================================================
# USER SETTINGS - edit this block
# ======================================================================================================

TASK = "segmentation"  # "detection" or "segmentation"; "pose", "keypoint", "classification" are extension points
BENCHMARK = True  # True: evaluate every model in MODELS and draw the comparison chart. False: SINGLE_MODEL only

# Trial setup: pretrained nano models on the small Ultralytics sample datasets (4 test images each).
# Replace the weights and data paths below with your own trained models and test sets.
SAMPLES = r"C:\Users\DilshanR\Desktop\Thullex\datasets"
SAMPLE_YAMLS = r"C:\Users\DilshanR\AppData\Local\Programs\Python\Python312\Lib\site-packages\ultralytics\cfg\datasets"
ROBOFLOW_WEIGHTS = r"C:\Users\DilshanR\.roboflow\models"

COCO8_YAML = SAMPLE_YAMLS + r"\coco8.yaml"
COCO8_DIRS = {"images_dir": SAMPLES + r"\coco8\images\val", "labels_dir": SAMPLES + r"\coco8\labels\val"}
COCO8_SEG_YAML = SAMPLE_YAMLS + r"\coco8-seg.yaml"
COCO8_SEG_DIRS = {"images_dir": SAMPLES + r"\coco8-seg\images\val", "labels_dir": SAMPLES + r"\coco8-seg\labels\val"}

DETECTION_MODELS = [
    ModelSpec(
        name="YOLO11n",
        architecture="yolo",
        weights="yolo11n.pt",
        data_path=COCO8_YAML,
        data_format="yolo",
        options={"allow_download": True, **COCO8_DIRS},
    ),
    ModelSpec(
        name="RF-DETR Nano",
        architecture="rfdetr",
        weights=ROBOFLOW_WEIGHTS + r"\rf-detr-nano.pth",
        data_path=COCO8_YAML,
        data_format="yolo",
        options={"variant": "RFDETRNano", **COCO8_DIRS},
    ),
]

SEGMENTATION_MODELS = [
    ModelSpec(
        name="YOLO11n-seg",
        architecture="yolo",
        weights="yolo11n-seg.pt",
        data_path=COCO8_SEG_YAML,
        data_format="yolo",
        options={"allow_download": True, **COCO8_SEG_DIRS},
    ),
    ModelSpec(
        name="RF-DETR Seg Nano",
        architecture="rfdetr",
        weights=ROBOFLOW_WEIGHTS + r"\rf-detr-seg-nano.pt",
        data_path=COCO8_SEG_YAML,
        data_format="yolo",
        options={"variant": "RFDETRSegNano", **COCO8_SEG_DIRS},
    ),
]

MODELS = SEGMENTATION_MODELS if TASK == "segmentation" else DETECTION_MODELS  # follows TASK automatically

SINGLE_MODEL = MODELS[0].name  # name of the model to run when BENCHMARK is False

DEVICE = "auto"  # "auto", "cpu", "cuda", "cuda:0"
CONF_THRESHOLD = 0.25  # operating confidence: used for precision / recall / F1 and for the timed pass
NMS_IOU = 0.7  # NMS IoU for architectures that use NMS (YOLO); RF-DETR is NMS-free
WARMUP_RUNS = 5  # untimed predictions before timing starts
MAX_IMAGES = None  # e.g. 50 for a quick trial run; None uses the whole test set
OUTPUT_DIR = "results"
SHOW_PLOT = True  # open the saved benchmark.png in the default image viewer

# ======================================================================================================
# TERMINAL OUTPUT
# ======================================================================================================

# One colour per model, in MODELS order; the benchmark chart uses the same ones.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


class Console:
    """Coloured, boxed terminal output.

    Colours are used only on an interactive terminal (NO_COLOR turns them off, FORCE_COLOR turns them on);
    box-drawing characters fall back to plain ASCII when the output encoding cannot show them, so piping the
    output to a file still gives a clean text report.
    """

    CODES = {"bold": "1", "dim": "2", "red": "91", "green": "92", "yellow": "93", "cyan": "96"}
    UNICODE = {"h": "─", "v": "│", "tl": "┌", "tm": "┬", "tr": "┐", "ml": "├", "mm": "┼", "mr": "┤",
               "bl": "└", "bm": "┴", "br": "┘", "dot": "●", "full": "█", "empty": "░", "arrow": "→", "check": "✓"}  # fmt: skip
    ASCII = {"h": "-", "v": "|", "tl": "+", "tm": "+", "tr": "+", "ml": "+", "mm": "+", "mr": "+",
             "bl": "+", "bm": "+", "br": "+", "dot": "*", "full": "#", "empty": ".", "arrow": "->", "check": "OK"}  # fmt: skip

    def __init__(self):
        self.interactive = sys.stdout.isatty()
        if "NO_COLOR" in os.environ:
            self.color = False
        elif "FORCE_COLOR" in os.environ:
            self.color = True
        else:
            self.color = self.interactive and self._enable_ansi()
        try:
            "".join(self.UNICODE.values()).encode(sys.stdout.encoding or "ascii")
            self.box = self.UNICODE
        except (UnicodeEncodeError, LookupError):
            self.box = self.ASCII

    @staticmethod
    def _enable_ansi() -> bool:
        """Turn on ANSI escape handling in the Windows console; other platforms have it already."""
        if sys.platform != "win32":
            return True
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
        except Exception:
            return False

    def paint(self, text: str, *styles: str) -> str:
        """Wrap text in the given styles: names from CODES, or a '#rrggbb' colour."""
        if not self.color or not styles:
            return text
        codes = []
        for style in styles:
            if style.startswith("#"):
                codes.append(f"38;2;{int(style[1:3], 16)};{int(style[3:5], 16)};{int(style[5:7], 16)}")
            else:
                codes.append(self.CODES[style])
        return f"\033[{';'.join(codes)}m{text}\033[0m"

    def title(self, text: str) -> None:
        print()
        print(self.paint(f" {text.upper()} ", "bold", "cyan"))
        print(self.paint(self.box["h"] * (len(text) + 2), "cyan"))

    def fields(self, pairs: list[tuple[str, str]]) -> None:
        width = max(len(label) for label, _ in pairs)
        for label, value in pairs:
            print(f"  {self.paint(label.ljust(width), 'dim')}  {value}")

    def model_heading(self, name: str, color: str, detail: str) -> None:
        print()
        print(f"{self.paint(self.box['dot'], color)} {self.paint(name, 'bold', color)}  {self.paint(detail, 'dim')}")

    def step(self, text: str) -> None:
        print(f"  {self.paint(text, 'dim')}")

    def progress(self, label: str, done: int, total: int, color: str | None = None) -> None:
        """A one-line progress bar, redrawn in place on a terminal and printed once at the end otherwise."""
        finished = done == total
        if not self.interactive and not finished:
            return
        slots = 24
        filled = slots * done // total
        bar = self.paint(self.box["full"] * filled, *([color] if color else [])) + self.paint(self.box["empty"] * (slots - filled), "dim")
        tail = self.paint(self.box["check"], "green") if finished else ""
        line = f"  {label:<24} {bar} {done:>{len(str(total))}}/{total} {tail}"
        print(("\r" if self.interactive else "") + line, end="\n" if finished else "", flush=True)

    def warn(self, text: str) -> None:
        print(f"{self.paint('Warning:', 'bold', 'yellow')} {self.paint(text, 'yellow')}")

    def note(self, text: str) -> None:
        print(f"{self.paint('Note:', 'bold', 'yellow')} {text}")

    def error(self, heading: str, text: str = "") -> None:
        print(f"\n{self.paint(heading, 'bold', 'red')} {text}".rstrip())

    def saved(self, path) -> None:
        print(f"  {self.paint(self.box['arrow'], 'green')} {path}")

    def table(self, header: list, sections: list[tuple[str, list[list]]], left: set[int]) -> None:
        """Print a boxed table. A cell is text or (text, *styles); columns in `left` are left-aligned."""

        def text(cell) -> str:
            return cell if isinstance(cell, str) else cell[0]

        def styles(cell) -> tuple:
            return () if isinstance(cell, str) else tuple(cell[1:])

        box = self.box
        rows = [header] + [row for _, section in sections for row in section]
        widths = [max(len(text(row[column])) for row in rows) for column in range(len(header))]
        widths[0] = max(widths[0], *(len(title) for title, _ in sections))

        def rule(start: str, middle: str, end: str) -> str:
            return self.paint(start + middle.join(box["h"] * (width + 2) for width in widths) + end, "dim")

        def line(cells: list) -> str:
            bar = self.paint(box["v"], "dim")
            parts = []
            for column, (cell, width) in enumerate(zip(cells, widths)):
                padded = text(cell).ljust(width) if column in left else text(cell).rjust(width)
                parts.append(f" {self.paint(padded, *styles(cell))} ")
            return bar + bar.join(parts) + bar

        print(rule(box["tl"], box["tm"], box["tr"]))
        print(line(header))
        for title, section in sections:
            print(rule(box["ml"], box["mm"], box["mr"]))
            print(line([(title, "bold", "cyan")] + [""] * (len(header) - 1)))
            for row in section:
                print(line(row))
        print(rule(box["bl"], box["bm"], box["br"]))


console = Console()

# ======================================================================================================
# DEPENDENCY CHECK
# ======================================================================================================


@dataclass(frozen=True)
class Requirement:
    distributions: tuple[str, ...]  # any one of these pip distributions satisfies the requirement
    minimum: str
    needed_for: str  # "core", "plot", or an architecture name


REQUIRED = [
    Requirement(("numpy",), "1.24.0", "core"),
    Requirement(("torch",), "2.0.0", "core"),
    Requirement(("supervision",), "0.26.0", "core"),
    Requirement(("opencv-python", "opencv-python-headless", "opencv-contrib-python"), "4.8.0", "core"),
    Requirement(("pyyaml",), "6.0", "core"),
    Requirement(("matplotlib",), "3.7.0", "plot"),
    Requirement(("ultralytics",), "8.3.0", "yolo"),
    Requirement(("rfdetr",), "1.2.0", "rfdetr"),
]


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version.split("+")[0])[:3])


def check_dependencies(architectures: set[str], need_plot: bool) -> None:
    """Exit with install instructions if a library needed for this run is missing or too old."""
    wanted = {"core"} | architectures | ({"plot"} if need_plot else set())
    problems = []
    for requirement in REQUIRED:
        if requirement.needed_for not in wanted:
            continue
        found = None
        for distribution in requirement.distributions:
            try:
                found = metadata.version(distribution)
                break
            except metadata.PackageNotFoundError:
                continue
        package = requirement.distributions[0]
        if found is None:
            problems.append((package, requirement.minimum, "not installed"))
        elif _version_tuple(found) < _version_tuple(requirement.minimum):
            problems.append((package, requirement.minimum, f"found {found}"))

    if not problems:
        return

    console.error("Some required libraries are missing or too old:")
    print()
    console.table(
        ["Library", "Required", "Status"],
        [("Missing or outdated", [[package, f">= {minimum}", (status, "red")] for package, minimum, status in problems])],
        left={0, 1, 2},
    )
    print("\nInstall them with:\n")
    command = "pip install " + " ".join(f'"{package}>={minimum}"' for package, minimum, _ in problems)
    print("  " + console.paint(command, "bold", "green"))
    if any(package == "torch" for package, _, _ in problems):
        print("\n  For GPU support pick the torch build for your CUDA version: https://pytorch.org/get-started/locally/")
    print()
    sys.exit(1)


# ======================================================================================================
# REGISTRIES (extension points)
# ======================================================================================================

ARCHITECTURES: dict[str, type[ModelAdapter]] = {}
DATA_FORMATS: dict[str, Callable[..., Any]] = {}  # loader(spec, masks=False) -> dataset
TASKS: dict[str, type[TaskEvaluator]] = {}


def register_architecture(name: str):
    def decorator(cls):
        ARCHITECTURES[name] = cls
        return cls

    return decorator


def register_data_format(name: str):
    def decorator(function):
        DATA_FORMATS[name] = function
        return function

    return decorator


def register_task(name: str):
    def decorator(cls):
        TASKS[name] = cls
        return cls

    return decorator


class ConfigError(Exception):
    """A problem with the settings that the user can fix (bad path, unknown name, unsupported task)."""


# ======================================================================================================
# MODEL ADAPTERS
# ======================================================================================================


class ModelAdapter(ABC):
    """Wraps one architecture behind a common interface.

    predict() is the only call that is timed, so anything that should not count as inference time
    (colour conversion, converting the output to supervision objects) belongs in prepare() / to_output().
    """

    supported_tasks: set[str] = set()

    def __init__(self, spec: ModelSpec, task: str, device: str, nms_iou: float):
        self.spec = spec
        self.task = task
        self.device = device
        self.nms_iou = nms_iou
        self.model = None
        self.color: str | None = None  # terminal colour for this model's progress bar, set by the Evaluator

    @abstractmethod
    def load(self) -> None:
        """Load the weights onto self.device."""

    @property
    @abstractmethod
    def class_names(self) -> dict[int, str]:
        """Class id -> class name as the model outputs them."""

    def prepare(self, image_bgr):
        """Untimed: turn a BGR uint8 image into whatever predict() wants."""
        return image_bgr

    @abstractmethod
    def predict(self, prepared, conf: float):
        """Timed: run the model on one image and return its raw output."""

    @abstractmethod
    def to_output(self, raw):
        """Untimed: convert the raw output to the task's common type (sv.Detections for detection)."""

    def forward_ms(self, raw) -> float | None:
        """Model-forward-only time for the last prediction, if the library reports one."""
        return None

    def sync(self) -> None:
        """Wait for pending GPU work so wall-clock timing is accurate."""
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()

    def _weights_path(self) -> str:
        path = Path(self.spec.weights)
        if path.is_file():
            return str(path)
        if self.spec.options.get("allow_download"):
            return self.spec.weights
        raise ConfigError(f"[{self.spec.name}] weights file not found: {self.spec.weights}")


@register_architecture("yolo")
class YoloAdapter(ModelAdapter):
    supported_tasks = {"detection", "segmentation", "pose", "keypoint", "classification"}

    def load(self) -> None:
        from ultralytics import YOLO

        self.model = YOLO(self._weights_path())

    @property
    def class_names(self) -> dict[int, str]:
        override = self.spec.options.get("class_names")
        if override:
            return dict(enumerate(override))
        return {int(index): str(name) for index, name in self.model.names.items()}

    def predict(self, prepared, conf: float):
        kwargs = {"conf": conf, "iou": self.nms_iou, "device": self.device, "verbose": False}
        if self.spec.options.get("imgsz"):
            kwargs["imgsz"] = self.spec.options["imgsz"]
        return self.model.predict(prepared, **kwargs)[0]

    def to_output(self, raw):
        import supervision as sv

        return sv.Detections.from_ultralytics(raw)

    def forward_ms(self, raw) -> float | None:
        return float(raw.speed["inference"])


@register_architecture("rfdetr")
class RFDETRAdapter(ModelAdapter):
    supported_tasks = {"detection", "segmentation", "keypoint"}

    def load(self) -> None:
        import rfdetr

        options = self.spec.options
        kwargs = {"device": self.device}
        if options.get("trust_checkpoint"):
            kwargs["trust_checkpoint"] = True
        variant = options.get("variant")
        if variant:
            if not hasattr(rfdetr, variant):
                raise ConfigError(f"[{self.spec.name}] unknown RF-DETR variant '{variant}'")
            if Path(self.spec.weights).is_file() or not options.get("allow_download"):
                kwargs["pretrain_weights"] = self._weights_path()
            self.model = getattr(rfdetr, variant)(**kwargs)
        else:
            self.model = rfdetr.from_checkpoint(self._weights_path(), **kwargs)
        if options.get("optimize"):
            self.model.optimize_for_inference()

    @property
    def class_names(self) -> dict[int, str]:
        names = self.spec.options.get("class_names") or self.model.class_names
        return dict(enumerate(names))

    def prepare(self, image_bgr):
        import cv2

        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # RF-DETR expects RGB

    def predict(self, prepared, conf: float):
        kwargs = {"threshold": conf, "include_source_image": False}
        resolution = self.spec.options.get("resolution")
        if resolution:
            kwargs["shape"] = (resolution, resolution)
        return self.model.predict(prepared, **kwargs)

    def to_output(self, raw):
        return raw  # already sv.Detections


# ======================================================================================================
# DATASET LOADERS
# ======================================================================================================


def _existing_dir(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


@register_data_format("yolo")
def load_yolo_dataset(spec: ModelSpec, masks: bool = False):
    """YOLO layout: data.yaml + images/ + labels/ (one .txt per image). masks=True reads the polygons as masks."""
    import supervision as sv
    import yaml

    yaml_path = Path(spec.data_path)
    if yaml_path.is_dir():
        yaml_path = yaml_path / "data.yaml"
    if not yaml_path.is_file():
        raise ConfigError(f"[{spec.name}] YOLO data.yaml not found: {yaml_path}")
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    images_dir = spec.options.get("images_dir")
    if images_dir:
        images_dir = Path(images_dir)
    else:
        entry = config.get(spec.split)
        if not isinstance(entry, str):
            raise ConfigError(
                f"[{spec.name}] data.yaml has no '{spec.split}' folder entry. Set split= on the ModelSpec "
                f"or options={{'images_dir': ..., 'labels_dir': ...}}."
            )
        root = Path(config.get("path") or ".")
        if not root.is_absolute():
            root = yaml_path.parent / root
        # Roboflow exports write "../test/images" relative to a folder that does not exist; try the plain form too.
        images_dir = _existing_dir(
            [Path(entry), root / entry, yaml_path.parent / entry, yaml_path.parent / entry.lstrip("./\\")]
        )
        if images_dir is None:
            raise ConfigError(f"[{spec.name}] could not find the '{spec.split}' images folder '{entry}' from {yaml_path}")

    labels_dir = spec.options.get("labels_dir")
    if labels_dir:
        labels_dir = Path(labels_dir)
    else:
        parts = list(Path(images_dir).parts)
        positions = [index for index, part in enumerate(parts) if part == "images"]
        if not positions:
            raise ConfigError(f"[{spec.name}] cannot derive the labels folder from {images_dir}; set options['labels_dir']")
        parts[positions[-1]] = "labels"
        labels_dir = Path(*parts)
    for folder in (images_dir, labels_dir):
        if not Path(folder).is_dir():
            raise ConfigError(f"[{spec.name}] folder not found: {folder}")

    # supervision opens data.yaml with the system encoding, which fails on Windows for UTF-8 files with
    # non-ASCII characters. Hand it an ASCII-only copy holding just the class names.
    names = config.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    if not names:
        raise ConfigError(f"[{spec.name}] data.yaml has no 'names' list: {yaml_path}")
    with tempfile.TemporaryDirectory() as folder:
        clean_yaml = Path(folder) / "data.yaml"
        clean_yaml.write_text(yaml.safe_dump({"nc": len(names), "names": [str(name) for name in names]}), encoding="ascii")
        return sv.DetectionDataset.from_yolo(
            images_directory_path=str(images_dir),
            annotations_directory_path=str(labels_dir),
            data_yaml_path=str(clean_yaml),
            force_masks=masks,
        )


@register_data_format("coco")
def load_coco_dataset(spec: ModelSpec, masks: bool = False):
    """COCO layout: one annotations .json, images next to it unless options['images_dir'] says otherwise."""
    import supervision as sv

    path = Path(spec.data_path)
    if path.is_dir():
        candidates = [path / "_annotations.coco.json", path / spec.split / "_annotations.coco.json"]
        annotations = next((candidate for candidate in candidates if candidate.is_file()), None)
        if annotations is None:
            raise ConfigError(f"[{spec.name}] no _annotations.coco.json in {path} or {path / spec.split}")
    elif path.is_file():
        annotations = path
    else:
        raise ConfigError(f"[{spec.name}] COCO annotations not found: {path}")

    images_dir = Path(spec.options.get("images_dir") or annotations.parent)
    if not images_dir.is_dir():
        raise ConfigError(f"[{spec.name}] images folder not found: {images_dir}")
    return sv.DetectionDataset.from_coco(
        images_directory_path=str(images_dir), annotations_path=str(annotations), force_masks=masks
    )


# ======================================================================================================
# RESULTS
# ======================================================================================================


@dataclass
class EvalResult:
    model_name: str
    architecture: str
    task: str
    device: str
    num_images: int
    metrics: dict[str, float]  # accuracy metrics, keyed as in TaskEvaluator.metric_labels
    timing: dict[str, float]  # avg_ms, median_ms, p95_ms, min_ms, max_ms, fps, forward_avg_ms (if known)
    image_names: list[str]
    per_image_ms: list[float]
    per_image_forward_ms: list[float | None]
    notes: list[str] = field(default_factory=list)


TIMING_LABELS = {
    "avg_ms": "Avg inference (ms)",
    "median_ms": "Median inference (ms)",
    "p95_ms": "p95 inference (ms)",
    "min_ms": "Min inference (ms)",
    "max_ms": "Max inference (ms)",
    "fps": "Throughput (FPS)",
    "forward_avg_ms": "Avg forward only (ms)",
}
HIGHER_IS_BETTER_TIMING = {"fps"}


def summarise_times(times_ms: list[float], forward_ms: list[float | None]) -> dict[str, float]:
    import numpy as np

    times = np.asarray(times_ms, dtype=float)
    timing = {
        "avg_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "fps": float(1000.0 / times.mean()),
    }
    known = [value for value in forward_ms if value is not None]
    if known:
        timing["forward_avg_ms"] = float(np.mean(known))
    return timing


# ======================================================================================================
# TASK EVALUATORS
# ======================================================================================================


class TaskEvaluator(ABC):
    """Scores one model on one dataset for one task.

    The timing loop is shared; a task only decides how the dataset is loaded and how predictions are scored.
    """

    metric_labels: dict[str, str] = {}  # metric key -> display label, in display order (all higher-is-better)
    headline_metric: str = ""  # the metric used for the speed-vs-accuracy view
    needs_masks: bool = False  # True makes the data loaders read mask annotations

    def __init__(self, conf_threshold: float, warmup_runs: int, max_images: int | None):
        self.conf_threshold = conf_threshold
        self.warmup_runs = warmup_runs
        self.max_images = max_images

    def load_dataset(self, spec: ModelSpec):
        if spec.data_format not in DATA_FORMATS:
            raise ConfigError(
                f"[{spec.name}] unknown data format '{spec.data_format}'. Available: {', '.join(sorted(DATA_FORMATS))}"
            )
        return DATA_FORMATS[spec.data_format](spec, masks=self.needs_masks)

    def prepare_output(self, output, adapter: ModelAdapter):
        """Untimed, called once per prediction as it is produced; a task can validate or compact it here."""
        return output

    @abstractmethod
    def evaluate(self, adapter: ModelAdapter, dataset) -> EvalResult:
        ...

    def _run_pass(self, adapter: ModelAdapter, image_paths: list[str], conf: float, label: str, timed: bool):
        """Run the model over every image at batch size 1. Returns (outputs, times_ms, forward_ms)."""
        import cv2

        outputs, times_ms, forward_ms = [], [], []
        total = len(image_paths)
        step = max(1, total // 50)
        for index, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            if image is None:
                raise ConfigError(f"[{adapter.spec.name}] could not read image: {image_path}")
            prepared = adapter.prepare(image)
            if timed and index == 0:
                for _ in range(self.warmup_runs):
                    adapter.predict(prepared, conf)
            adapter.sync()
            start = time.perf_counter()
            raw = adapter.predict(prepared, conf)
            adapter.sync()
            times_ms.append((time.perf_counter() - start) * 1000.0)
            forward_ms.append(adapter.forward_ms(raw))
            outputs.append(self.prepare_output(adapter.to_output(raw), adapter))
            if (index + 1) % step == 0 or index + 1 == total:
                console.progress(label, index + 1, total, adapter.color)
        return outputs, times_ms, forward_ms


class ClassAligner:
    """Maps a model's class ids onto the dataset's class ids by class name.

    Exports of the same dataset often number classes differently (COCO exports commonly shift ids by one),
    so matching ids directly would silently score a correct model as wrong.
    """

    def __init__(self, model_names: dict[int, str], dataset_classes: list[str]):
        self.model_names = model_names
        self.lookup = {self._key(name): index for index, name in enumerate(dataset_classes)}
        self.dropped = 0
        self.unmatched: set[str] = set()
        matched = [name for name in model_names.values() if self._key(name) in self.lookup]
        self.identity = not matched  # no name in common: fall back to matching raw ids
        self.missing_in_model = sorted(
            name for name in dataset_classes if self._key(name) not in {self._key(n) for n in model_names.values()}
        )

    @staticmethod
    def _key(name) -> str:
        return str(name).strip().lower()

    def __call__(self, detections):
        import numpy as np

        if self.identity or detections.class_id is None or len(detections) == 0:
            return detections
        names = detections.data.get("class_name")
        if names is None or len(names) != len(detections):
            names = [self.model_names.get(int(class_id), str(class_id)) for class_id in detections.class_id]
        mapped = np.array([self.lookup.get(self._key(name), -1) for name in names], dtype=int)
        keep = mapped >= 0
        self.dropped += int((~keep).sum())
        self.unmatched.update(str(name) for name, kept in zip(names, keep) if not kept)
        detections = detections[keep]
        detections.class_id = mapped[keep]
        return detections

    def notes(self) -> list[str]:
        notes = []
        if self.identity:
            notes.append("No class names in common between model and dataset: class ids were matched as-is.")
            return notes
        if self.dropped:
            notes.append(
                f"{self.dropped} predictions dropped for classes not in the dataset: {', '.join(sorted(self.unmatched))}."
            )
        if self.missing_in_model:
            notes.append(f"Dataset classes the model does not predict: {', '.join(self.missing_in_model)}.")
        return notes


@register_task("detection")
class DetectionEvaluator(TaskEvaluator):
    metric_labels = {
        "map50_95": "mAP50-95",
        "map50": "mAP50",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
    }
    headline_metric = "map50_95"
    task_name = "detection"
    metric_target = "boxes"  # a supervision MetricTarget value: what IoU is computed on
    MAP_CONF = 0.001  # mAP integrates over the whole confidence range, so it needs low-confidence predictions

    def extra_metrics(self, dense: list, targets: list) -> dict[str, float]:
        """Additional metrics from the low-confidence predictions; subclasses add theirs here."""
        return {}

    def evaluate(self, adapter: ModelAdapter, dataset) -> EvalResult:
        from supervision.metrics import AveragingMethod, F1Score, MeanAveragePrecision, MetricTarget, Precision, Recall

        target = MetricTarget(self.metric_target)

        image_paths = list(dataset.image_paths)
        if self.max_images:
            image_paths = image_paths[: self.max_images]
        if not image_paths:
            raise ConfigError(f"[{adapter.spec.name}] the test set contains no images")
        targets = [dataset.annotations[path] for path in image_paths]
        aligner = ClassAligner(adapter.class_names, list(dataset.classes))

        # Pass 1, timed, at the operating confidence: what deployment would see. Gives precision / recall / F1.
        outputs, times_ms, forward_ms = self._run_pass(
            adapter, image_paths, self.conf_threshold, f"timed pass (conf {self.conf_threshold})", timed=True
        )
        operating = [aligner(output) for output in outputs]
        # Pass 2, untimed, at a very low confidence for mAP.
        outputs, _, _ = self._run_pass(adapter, image_paths, self.MAP_CONF, "mAP pass", timed=False)
        dense = [aligner(output) for output in outputs]

        map_result = MeanAveragePrecision(metric_target=target).update(dense, targets).compute()
        scoring = {"metric_target": target, "averaging_method": AveragingMethod.MACRO}
        metrics = {
            "map50_95": float(map_result.map50_95),
            "map50": float(map_result.map50),
            "precision": float(Precision(**scoring).update(operating, targets).compute().precision_at_50),
            "recall": float(Recall(**scoring).update(operating, targets).compute().recall_at_50),
            "f1": float(F1Score(**scoring).update(operating, targets).compute().f1_50),
        }
        metrics.update(self.extra_metrics(dense, targets))
        return EvalResult(
            model_name=adapter.spec.name,
            architecture=adapter.spec.architecture,
            task=self.task_name,
            device=adapter.device,
            num_images=len(image_paths),
            metrics=metrics,
            timing=summarise_times(times_ms, forward_ms),
            image_names=[Path(path).name for path in image_paths],
            per_image_ms=times_ms,
            per_image_forward_ms=forward_ms,
            notes=aligner.notes(),
        )


def _compact_masks(detections):
    """Swap dense (N, H, W) masks for supervision's run-length CompactMask so a whole test set fits in memory."""
    import numpy as np
    import supervision as sv

    mask = detections.mask
    if mask is None or len(detections) == 0 or isinstance(mask, sv.CompactMask):
        return detections
    detections.mask = sv.CompactMask.from_dense(np.asarray(mask, dtype=bool), detections.xyxy, image_shape=mask.shape[1:])
    return detections


@register_task("segmentation")
class SegmentationEvaluator(DetectionEvaluator):
    """Instance segmentation: the same loop as detection, with IoU computed on masks instead of boxes."""

    metric_labels = {
        "map50_95": "Mask mAP50-95",
        "map50": "Mask mAP50",
        "precision": "Mask precision",
        "recall": "Mask recall",
        "f1": "Mask F1",
        "box_map50_95": "Box mAP50-95",
    }
    task_name = "segmentation"
    metric_target = "masks"
    needs_masks = True

    def load_dataset(self, spec: ModelSpec):
        dataset = super().load_dataset(spec)
        for annotation in dataset.annotations.values():
            _compact_masks(annotation)
        return dataset

    def prepare_output(self, output, adapter: ModelAdapter):
        if len(output) > 0 and output.mask is None:
            raise ConfigError(
                f"[{adapter.spec.name}] the model returned boxes but no masks. Task 'segmentation' needs "
                f"segmentation weights (for example a YOLO '-seg' model or an RF-DETR Seg checkpoint)."
            )
        return _compact_masks(output)

    def extra_metrics(self, dense: list, targets: list) -> dict[str, float]:
        from supervision.metrics import MeanAveragePrecision

        return {"box_map50_95": float(MeanAveragePrecision().update(dense, targets).compute().map50_95)}


def _register_planned_task(name: str, hint: str) -> None:
    """Reserve a task name so selecting it gives a clear message instead of 'unknown task'."""

    @register_task(name)
    class PlannedTask(TaskEvaluator):
        def __init__(self, *args, **kwargs):
            raise ConfigError(
                f"Task '{name}' is not implemented yet. To add it, subclass TaskEvaluator, decorate it with "
                f"@register_task('{name}') and {hint}"
            )

        def evaluate(self, adapter, dataset):  # pragma: no cover - never reached
            raise NotImplementedError


_register_planned_task("pose", "score sv.KeyPoints predictions with OKS-based mAP against keypoint ground truth.")
_register_planned_task("keypoint", "score sv.KeyPoints predictions with OKS-based mAP against keypoint ground truth.")
_register_planned_task(
    "classification", "add a folder-per-class data format and report top-1 / top-5 accuracy from the model's probabilities."
)


# ======================================================================================================
# REPORTING
# ======================================================================================================


def _best_model(results: list[EvalResult], key: str, group: str, higher_is_better: bool) -> str:
    values = [(getattr(result, group).get(key), result.model_name) for result in results]
    values = [(value, name) for value, name in values if value is not None]
    if len(values) < 2:
        return ""
    pick = max if higher_is_better else min
    best = pick(value for value, _ in values)
    winners = [name for value, name in values if value == best]
    return winners[0] if len(winners) == 1 else "tie"


def summary_rows(results: list[EvalResult], metric_labels: dict[str, str]) -> list[list[str]]:
    """Rows of [label, value per model..., best model] shared by the console table and the chart."""
    rows = []
    for key, label in metric_labels.items():
        values = [f"{result.metrics[key]:.3f}" for result in results]
        rows.append([label, *values, _best_model(results, key, "metrics", True)])
    for key, label in TIMING_LABELS.items():
        if not any(key in result.timing for result in results):
            continue
        values = [f"{result.timing[key]:.1f}" if key in result.timing else "-" for result in results]
        rows.append([label, *values, _best_model(results, key, "timing", key in HIGHER_IS_BETTER_TIMING)])
    rows.append(["Images", *[str(result.num_images) for result in results], ""])
    rows.append(["Device", *[result.device for result in results], ""])
    return rows


def print_report(results: list[EvalResult], evaluator_cls: type[TaskEvaluator]) -> None:
    """Print the results table: accuracy, speed and run details, with the better value per row highlighted."""
    labels = evaluator_cls.metric_labels
    names = [result.model_name for result in results]
    colors = {name: SERIES_COLORS[index % len(SERIES_COLORS)] for index, name in enumerate(names)}
    compare = len(results) > 1

    def styled(row: list[str]) -> list:
        best = row[-1]
        cells: list = [row[0]]
        for name, value in zip(names, row[1:-1]):
            cells.append((value, "bold", "green") if best == name else value)
        if compare:
            cells.append((best, colors[best]) if best in colors else (best, "dim"))
        return cells

    rows = [styled(row) for row in summary_rows(results, labels)]
    header: list = [("Metric", "bold"), *[(name, "bold", colors[name]) for name in names]]
    if compare:
        header.append(("Best", "bold"))
    sections = [("Accuracy", rows[: len(labels)]), ("Speed", rows[len(labels) : -2]), ("Run", rows[-2:])]

    console.title("Results")
    console.table(header, sections, left={0, len(header) - 1} if compare else {0})

    if compare:
        key = evaluator_cls.headline_metric
        accurate = max(results, key=lambda result: result.metrics[key])
        fastest = min(results, key=lambda result: result.timing["avg_ms"])
        print()
        console.fields(
            [
                ("Most accurate", f"{console.paint(accurate.model_name, 'bold', colors[accurate.model_name])}  {labels[key]} {accurate.metrics[key]:.3f}"),
                ("Fastest", f"{console.paint(fastest.model_name, 'bold', colors[fastest.model_name])}  {fastest.timing['avg_ms']:.1f} ms per image ({fastest.timing['fps']:.1f} FPS)"),
            ]
        )
    notes = [(result.model_name, note) for result in results for note in result.notes]
    if notes:
        print()
    for name, note in notes:
        console.note(f"[{name}] {note}")


def save_results(results: list[EvalResult], run_dir: Path, settings: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "settings": settings,
        "models": [
            {key: value for key, value in asdict(result).items() if not key.startswith(("per_image", "image_names"))}
            for result in results
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(run_dir / "per_image_times.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "image", "inference_ms", "forward_only_ms"])
        for result in results:
            for name, total, forward in zip(result.image_names, result.per_image_ms, result.per_image_forward_ms):
                writer.writerow([result.model_name, name, f"{total:.3f}", "" if forward is None else f"{forward:.3f}"])


# ======================================================================================================
# BENCHMARK CHART
# ======================================================================================================

SURFACE, INK, INK_SECONDARY, INK_MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _style_axis(ax, title: str, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _grouped_bars(ax, groups: list[str], results: list[EvalResult], values: list[list[float]], fmt: str) -> None:
    """values[model][group]. One bar per model in each group, value printed on top."""
    import numpy as np

    count = len(results)
    width = 0.8 / count
    positions = np.arange(len(groups))
    top = max(max(row) for row in values)
    for index, (result, row) in enumerate(zip(results, values)):
        offsets = positions + (index - (count - 1) / 2) * width
        bars = ax.bar(offsets, row, width, color=SERIES_COLORS[index], edgecolor=SURFACE, linewidth=2, label=result.model_name)
        if count <= 4:
            for bar, value in zip(bars, row):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, value + top * 0.015, format(value, fmt),
                    ha="center", va="bottom", fontsize=8, color=INK_SECONDARY,
                )
    # With many groups the labels would run into each other, so break them over two lines.
    ax.set_xticks(positions, [group.replace(" ", "\n", 1) for group in groups] if len(groups) > 5 else groups)
    ax.set_ylim(0, top * 1.12)


def _verdict(results: list[EvalResult], evaluator_cls: type[TaskEvaluator]) -> str:
    key = evaluator_cls.headline_metric
    label = evaluator_cls.metric_labels[key]
    accurate = max(results, key=lambda result: result.metrics[key])
    fastest = min(results, key=lambda result: result.timing["avg_ms"])
    if accurate is fastest:
        return f"{accurate.model_name} is both the most accurate ({label} {accurate.metrics[key]:.3f}) and the fastest ({accurate.timing['avg_ms']:.1f} ms)"
    return (
        f"Most accurate: {accurate.model_name} ({label} {accurate.metrics[key]:.3f})   |   "
        f"Fastest: {fastest.model_name} ({fastest.timing['avg_ms']:.1f} ms per image)"
    )


def plot_benchmark(results: list[EvalResult], evaluator_cls: type[TaskEvaluator], run_dir: Path, show: bool) -> Path:
    import matplotlib

    matplotlib.use("Agg")  # render straight to the file; no Matplotlib window
    import matplotlib.pyplot as plt

    if len(results) > len(SERIES_COLORS):
        raise ConfigError(f"The benchmark chart supports up to {len(SERIES_COLORS)} models; got {len(results)}.")

    labels = evaluator_cls.metric_labels
    headline = evaluator_cls.headline_metric
    names = [result.model_name for result in results]

    fig = plt.figure(figsize=(14, 11.5), facecolor=SURFACE)
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.85], hspace=0.42, wspace=0.2, left=0.06, right=0.97, top=0.86, bottom=0.04)
    fig.suptitle(
        f"Model benchmark: {results[0].task}", x=0.06, y=0.975, ha="left", fontsize=16, fontweight="bold", color=INK
    )
    fig.text(0.06, 0.935, _verdict(results, evaluator_cls), fontsize=11, color=INK, ha="left")
    devices = sorted({result.device for result in results})
    context = f"{results[0].num_images} test images, batch size 1, device: {', '.join(devices)}"
    if len(devices) > 1:
        context += "   (models ran on different devices: timings are not directly comparable)"
    fig.text(0.06, 0.908, context, fontsize=9, color=INK_SECONDARY, ha="left")

    # Accuracy
    ax = fig.add_subplot(grid[0, 0])
    _style_axis(ax, "Accuracy (higher is better)")
    _grouped_bars(ax, list(labels.values()), results, [[result.metrics[key] for key in labels] for result in results], ".3f")

    # Inference time summary
    ax = fig.add_subplot(grid[0, 1])
    _style_axis(ax, "Inference time per image, ms (lower is better)")
    time_keys = ["avg_ms", "median_ms", "p95_ms"]
    _grouped_bars(ax, ["Average", "Median", "p95"], results, [[result.timing[key] for key in time_keys] for result in results], ".1f")

    # Per-image inference time distribution
    ax = fig.add_subplot(grid[1, 0])
    _style_axis(ax, "Per-image inference time distribution, ms", grid_axis="x")
    # Matplotlib 3.10 renamed vert=False to orientation="horizontal" and deprecated the old spelling.
    horizontal = {"orientation": "horizontal"} if _version_tuple(matplotlib.__version__) >= (3, 10) else {"vert": False}
    boxes = ax.boxplot(
        [result.per_image_ms for result in results][::-1], **horizontal, patch_artist=True, widths=0.5,
        medianprops={"color": INK, "linewidth": 1.5}, whiskerprops={"color": INK_MUTED}, capprops={"color": INK_MUTED},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": INK_MUTED, "markeredgecolor": "none", "alpha": 0.6},
    )
    for patch, color in zip(boxes["boxes"], SERIES_COLORS[: len(results)][::-1]):
        patch.set_facecolor(color)
        patch.set_edgecolor(SURFACE)
    # Names sit inside the plot, above each box, so long model names are never clipped at the figure edge.
    ax.set_yticks([])
    for position, name in enumerate(names[::-1], start=1):
        ax.text(0.01, position + 0.3, name, transform=ax.get_yaxis_transform(), fontsize=9, color=INK_SECONDARY, va="bottom")
    ax.set_ylim(0.5, len(results) + 0.7)
    ax.set_xlim(left=0)

    # Speed vs accuracy
    ax = fig.add_subplot(grid[1, 1])
    _style_axis(ax, "Speed vs accuracy (towards the top-left is better)", grid_axis="both")
    xs = [result.timing["avg_ms"] for result in results]
    ys = [result.metrics[headline] for result in results]
    for index, result in enumerate(results):
        ax.scatter(xs[index], ys[index], s=140, color=SERIES_COLORS[index], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(result.model_name, (xs[index], ys[index]), xytext=(10, 6), textcoords="offset points", fontsize=10, color=INK)
    x_pad = max(max(xs) - min(xs), max(xs) * 0.1) * 0.35
    y_pad = max(max(ys) - min(ys), 0.02) * 0.6
    ax.set_xlim(max(0, min(xs) - x_pad), max(xs) + x_pad * 2)  # extra room on the right for the point labels
    ax.set_ylim(max(0, min(ys) - y_pad), max(ys) + y_pad)
    ax.set_xlabel("Average inference time per image (ms)", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel(labels[headline], fontsize=9, color=INK_SECONDARY)

    # Summary table
    ax = fig.add_subplot(grid[2, :])
    ax.axis("off")
    ax.set_title("Summary", loc="left", fontsize=11, fontweight="bold", color=INK, pad=4)
    rows = summary_rows(results, labels)
    table = ax.table(cellText=rows, colLabels=["Metric", *names, "Best"], loc="upper center", cellLoc="left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURFACE)
        cell.get_text().set_color(INK if row == 0 or column == 0 else INK_SECONDARY)
        if row == 0:
            cell.get_text().set_fontweight("bold")
        elif 1 <= column <= len(results) and rows[row - 1][-1] == names[column - 1]:
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color(INK)

    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES_COLORS[index]) for index in range(len(results))]
    fig.legend(handles, names, loc="upper right", bbox_to_anchor=(0.97, 0.985), ncol=len(results), frameon=False, fontsize=10, labelcolor=INK)

    output = run_dir / "benchmark.png"
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    if show:
        open_file(output)
    return output


def open_file(path: Path) -> None:
    """Open a file in the system's default application (the image viewer for the chart)."""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))
        else:
            subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(path)], check=False)
    except OSError as error:
        print(f"Could not open {path} automatically ({error}); open it manually.")


# ======================================================================================================
# EVALUATOR
# ======================================================================================================


def _is_out_of_memory(error: Exception) -> bool:
    return "out of memory" in str(error).lower()


class Evaluator:
    """Evaluates one model, or benchmarks several, on a test set.

    Every argument defaults to the matching variable in the USER SETTINGS block.
    """

    def __init__(
        self,
        task: str = TASK,
        benchmark: bool = BENCHMARK,
        models: list[ModelSpec] | None = None,
        single_model: str | None = SINGLE_MODEL,
        device: str = DEVICE,
        conf_threshold: float = CONF_THRESHOLD,
        nms_iou: float = NMS_IOU,
        warmup_runs: int = WARMUP_RUNS,
        max_images: int | None = MAX_IMAGES,
        output_dir: str = OUTPUT_DIR,
        show_plot: bool = SHOW_PLOT,
    ):
        self.task = task.strip().lower()
        self.benchmark = benchmark
        self.models = list(MODELS if models is None else models)
        self.single_model = single_model
        self.device = device
        self.conf_threshold = conf_threshold
        self.nms_iou = nms_iou
        self.warmup_runs = warmup_runs
        self.max_images = max_images
        self.output_dir = Path(output_dir)
        self.show_plot = show_plot

    def _selected_models(self) -> list[ModelSpec]:
        if not self.models:
            raise ConfigError("No models configured. Add at least one ModelSpec to MODELS.")
        names = [spec.name for spec in self.models]
        if len(set(names)) != len(names):
            raise ConfigError(f"Model names must be unique; got {names}")
        if self.benchmark:
            if len(self.models) < 2:
                raise ConfigError("BENCHMARK is True but fewer than two models are configured.")
            return self.models
        if not self.single_model:
            return self.models[:1]
        for spec in self.models:
            if spec.name == self.single_model:
                return [spec]
        raise ConfigError(f"SINGLE_MODEL '{self.single_model}' is not in MODELS ({', '.join(names)})")

    def _validate(self, specs: list[ModelSpec]) -> None:
        if self.task not in TASKS:
            raise ConfigError(f"Unknown task '{self.task}'. Available: {', '.join(sorted(TASKS))}")
        for spec in specs:
            if spec.architecture not in ARCHITECTURES:
                raise ConfigError(
                    f"[{spec.name}] unknown architecture '{spec.architecture}'. Available: {', '.join(sorted(ARCHITECTURES))}"
                )
            if self.task not in ARCHITECTURES[spec.architecture].supported_tasks:
                raise ConfigError(f"[{spec.name}] architecture '{spec.architecture}' does not support task '{self.task}'")

    def _resolve_device(self) -> str:
        import torch

        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            console.warn("CUDA was requested but is not available; using the CPU.")
            return "cpu"
        return self.device

    def _evaluate_model(self, spec: ModelSpec, evaluator: TaskEvaluator, dataset, device: str, color: str) -> EvalResult:
        import torch

        adapter = ARCHITECTURES[spec.architecture](spec, self.task, device, self.nms_iou)
        adapter.color = color
        try:
            adapter.load()
            return evaluator.evaluate(adapter, dataset)
        except RuntimeError as error:
            if not (device.startswith("cuda") and _is_out_of_memory(error)):
                raise
        # The GPU ran out of memory: free it and repeat the whole evaluation on the CPU.
        print()
        console.warn(f"[{spec.name}] GPU out of memory; re-running on the CPU.")
        del adapter
        torch.cuda.empty_cache()
        adapter = ARCHITECTURES[spec.architecture](spec, self.task, "cpu", self.nms_iou)
        adapter.color = color
        adapter.load()
        result = evaluator.evaluate(adapter, dataset)
        result.notes.append("Ran on the CPU because the GPU ran out of memory.")
        return result

    def run(self) -> list[EvalResult]:
        specs = self._selected_models()
        check_dependencies({spec.architecture for spec in specs}, need_plot=self.benchmark)
        self._validate(specs)
        evaluator_cls = TASKS[self.task]
        evaluator = evaluator_cls(self.conf_threshold, self.warmup_runs, self.max_images)
        device = self._resolve_device()

        console.title("Model benchmark" if self.benchmark else "Model evaluation")
        console.fields(
            [
                ("Task", self.task),
                ("Device", device),
                ("Models", ", ".join(spec.name for spec in specs)),
                ("Confidence", str(self.conf_threshold)),
                ("Images", "all" if not self.max_images else f"first {self.max_images}"),
            ]
        )
        datasets: dict[tuple, Any] = {}
        results = []
        for index, spec in enumerate(specs):
            color = SERIES_COLORS[index % len(SERIES_COLORS)]
            console.model_heading(spec.name, color, f"{spec.architecture}  {spec.weights}")
            key = (spec.data_format, spec.data_path, spec.split, json.dumps(spec.options, sort_keys=True, default=str))
            if key not in datasets:
                console.step(f"loading {spec.data_format} test data: {spec.data_path}")
                datasets[key] = evaluator.load_dataset(spec)
            results.append(self._evaluate_model(spec, evaluator, datasets[key], device, color))

        print_report(results, evaluator_cls)
        if len({result.num_images for result in results}) > 1:
            counts = ", ".join(f"{result.model_name}: {result.num_images}" for result in results)
            console.warn(f"the models were scored on different numbers of images ({counts}).")
        if len({result.device for result in results}) > 1:
            console.warn("the models ran on different devices, so their timings are not directly comparable.")

        run_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        settings = {
            "task": self.task,
            "benchmark": self.benchmark,
            "conf_threshold": self.conf_threshold,
            "nms_iou": self.nms_iou,
            "warmup_runs": self.warmup_runs,
            "max_images": self.max_images,
            "models": [asdict(spec) for spec in specs],
        }
        save_results(results, run_dir, settings)
        console.title("Saved files")
        console.saved(run_dir / "summary.json")
        console.saved(run_dir / "per_image_times.csv")
        if self.benchmark:
            console.saved(plot_benchmark(results, evaluator_cls, run_dir, self.show_plot))
        print()
        return results


def main() -> int:
    try:
        Evaluator().run()
    except ConfigError as error:
        console.error("Configuration problem:", str(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
