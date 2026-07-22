# Setup

Getting Sentinel running locally with VS Code and Jupyter notebooks.

## 1. VS Code extensions

Install both:

- **Python** (Microsoft)
- **Jupyter** (Microsoft)

## 2. Environment

```bash
git clone <your-repo> sentinel && cd sentinel
python -m venv .venv

# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt`:

```
ultralytics
opencv-python
numpy
pyyaml
ollama
matplotlib
ipykernel
```

`ipykernel` must be installed **inside** the venv — that's what lets VS Code
offer it as a notebook kernel.

## 3. Point VS Code at the venv

- `Ctrl/Cmd + Shift + P` → **Python: Select Interpreter** → pick `./.venv`
- Open a notebook → **Select Kernel** (top right) → **Python Environments** → `.venv`

If your venv doesn't appear in the kernel list, it's almost always because
`ipykernel` isn't installed in it.

## 4. Verify GPU before anything else

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

If you have an NVIDIA card and this prints `False`, install the CUDA build of
PyTorch from [pytorch.org](https://pytorch.org) before continuing. Running on
CPU without realizing it is the single most common way to lose a day here.

Apple Silicon: check for `mps` instead — Ultralytics will use it if available.

## 5. YOLO weights

Nothing to download manually; Ultralytics fetches weights on first use.

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")     # or yolo26n.pt
```

Start with the nano variant. Move up to `s` or `m` only if accuracy on your
footage is genuinely the bottleneck.

## 6. Ollama

Install from [ollama.com](https://ollama.com), then pull a model sized to your
hardware:

```bash
ollama pull qwen3:8b        # decent GPU
ollama pull qwen3.5:4b      # CPU-only or modest laptop
ollama pull phi4-mini       # alternative small option

ollama serve                # usually already running as a service
```

Verify it's reachable:

```bash
curl http://localhost:11434/api/tags
```

## 7. Test video

Drop a clip in `data/raw/`. Use footage that resembles what you actually care
about — a fixed camera angle, realistic lighting, real compression. Clean
tutorial videos will make everything look like it works.

---

## Notebook workflow

Put this at the top of every notebook:

```python
%load_ext autoreload
%autoreload 2

import sys; sys.path.append("..")
from sentinel import detect, zones, store
```

`autoreload` picks up edits to `.py` modules on the next cell run, no kernel
restart. That's what makes the module-plus-notebook split pleasant instead of
annoying — edit a module in one tab, rerun a cell in another.

### Displaying frames

Do **not** use `cv2.imshow()` in a notebook. It opens a window that blocks the
kernel and frequently hangs. Show frames inline instead:

```python
import cv2
import matplotlib.pyplot as plt

def show(frame):
    plt.figure(figsize=(12, 7))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # OpenCV is BGR
    plt.axis("off")
    plt.show()
```

To review a whole processed clip, write an annotated `.mp4` to disk and open it
in VS Code's built-in video preview rather than trying to play video in a cell.

### Long-running cells

Processing a full video can take minutes. Wrap the frame loop in `tqdm` so you
can tell the difference between "working" and "hung":

```python
from tqdm.notebook import tqdm
for frame in tqdm(reader, total=reader.frame_count):
    ...
```

---

## Suggested `.gitignore`

```
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
data/raw/
data/thumbs/
*.db
*.pt
runs/
```

Notebook outputs bloat diffs badly on a public repo — consider clearing outputs
before committing, or adding [nbstripout](https://github.com/kynan/nbstripout).
`runs/` is where Ultralytics dumps its own output by default.