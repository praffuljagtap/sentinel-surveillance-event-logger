# Sentinel — Automated Surveillance Event Logger

Turn surveillance video into a readable event log.

A detection-and-tracking layer watches the footage and emits **structured
events** — people crossing a line, a door opening, an object leaving its place.
A small **local LLM** then reads those events and writes plain-English
summaries, and answers questions like *"was anyone near the back door after
9 PM?"*

Everything runs locally. No cloud, no API keys, no footage leaving the machine.

**Scope:** recorded video files only. The same pipeline would be useful pointed
at a live camera feed, but live ingestion is not supported here — offline
processing is far easier to debug and reason about.

See [SETUP.md](SETUP.md) to get running.

---

## Design principle

> **The LLM never looks at pixels.**

Vision does the perceiving. Language does the narrating. Keeping that boundary
clean is what makes the system fast, cheap, and — most importantly — honest: the
LLM can only describe events the detector actually produced.

```
video ──▶ YOLO + tracker ──▶ event rules ──▶ SQLite ──▶ local LLM ──▶ summaries
          (per frame)        (per track)    (ground     (per batch)   & Q&A
                                             truth)
```

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Where the whole CV/LLM ecosystem lives |
| Video I/O, ROI logic | OpenCV | Reading files, frame diffing, annotation |
| Detection + tracking | Ultralytics YOLO (YOLO11 or YOLO26) | Pretrained COCO classes + ByteTrack built in |
| Storage | SQLite | Single file, no server, ships with Python |
| Local LLM | Ollama | Local HTTP API; swap models without code changes |

**Model sizing.** Start with a `n` (nano) or `s` (small) YOLO variant — swapping
sizes is a one-line change. For the LLM, an 8B-class model if you have a decent
GPU, or a 4B-class model (Qwen3.5 4B, Phi-4-mini) on CPU-only hardware.
Summarizing JSON is not a hard language task; don't overspend here.

---

## Project structure

Logic lives in `.py` modules. Notebooks are **drivers** — they import the
modules, run them on a clip, and show you the result. This is the most important
structural decision in the project: it means the code survives outside the
notebook, stays diffable in git, and doesn't rot into 200 cells of copy-pasted
state.

```
sentinel/
├── README.md
├── SETUP.md
├── requirements.txt
├── config.yaml              # per-video zone definitions and thresholds
│
├── sentinel/                # the library
│   ├── __init__.py
│   ├── video.py             # frame reader, frame skipping
│   ├── detect.py            # YOLO load + track() wrapper
│   ├── zones.py             # line-crossing tests, ROI definitions
│   ├── events.py            # the event rules
│   ├── store.py             # SQLite writes/queries + thumbnail saving
│   └── narrate.py           # Ollama calls: summarize + answer questions
│
├── notebooks/
│   ├── calibrate.ipynb      # inspect footage, define lines/ROIs, tune thresholds
│   └── run.ipynb            # run the pipeline end to end, review the event log
│
├── data/
│   ├── raw/                 # input videos (gitignored)
│   └── thumbs/              # event thumbnails (gitignored)
│
└── sentinel.db              # SQLite event store (gitignored)
```

Two notebooks is enough. One is interactive work that genuinely needs a
notebook — looking at frames, clicking to place a line, watching a threshold
change the output. The other runs the pipeline and inspects results. Anything
else belongs in a module.

---

## Objectives

Roughly in dependency order — each one builds on the last, and each leaves you
with something that runs end to end.

**1. Read and detect.**
Decode a video, run YOLO on frames, draw boxes, write an annotated output file.
Includes frame skipping (every Nth frame — people don't teleport) since
full-rate HD detection is slow on CPU. Success: you can watch an annotated clip
and see correct labels.

**2. Track with persistent identity.**
Move from `predict()` to `track()` so each object carries a stable ID across
frames. This is the foundation everything else sits on — without it you have
"a person is visible," not "the same person moved." Success: IDs stay attached
through brief occlusions.

**3. Define zones per video.**
A config format for naming lines (for crossings) and rectangular ROIs (for doors
and monitored objects), plus the notebook that lets you draw them on a real
frame. Zones are per-camera-angle, so they need to be data, not code. Success:
a new video needs a config entry, not a code change.

**4. Detect people crossing.**
Log an event with direction when a tracked centroid crosses a named line.
Counting unique track IDs means someone loitering on the line isn't counted
twice. This is the easiest and most reliable event type — get it working first.

**5. Detect door state.**
Don't try to detect "a door" as an object. Watch an ROI for change against a
reference frame; sustained change means open, return to reference means closed.
Threshold is calibrated once per video. Slightly hacky, robust in practice.

**6. Detect objects moved.**
The hard one. Keep a reference crop of the monitored region and compare against
it **only when no person track overlaps that ROI**. That gating is the whole
trick — without it, someone walking past makes the object "disappear" and you
get constant false triggers.

**7. Persist events.**
Every event written to SQLite with a thumbnail of the triggering frame. The
thumbnails matter more than they sound: they're how you actually debug false
positives, by eyeballing what fired instead of trusting the log.

**8. Narrate.**
Batch events by time window, hand the LLM clean JSON, get back summaries and a
natural-language query path over the event table. Last, deliberately — a great
narrator on top of a flaky event stream just produces fluent nonsense.

---

## Guardrails

**Ground the LLM.** If the detector says `person`, the LLM must never upgrade
that to "delivery driver in a red jacket." Small models hallucinate exactly
these plausible details, and in anything surveillance-flavored a confident wrong
log is worse than no log. Prompt it to summarize *only* the supplied events and
to cite the events each sentence came from. Use structured/JSON output mode
rather than parsing free text.

**Never in the hot path.** The LLM runs periodically over batches of events.
Never per frame.

**Test on realistic footage from the start.** Surveillance angles, low light,
and compression artifacts behave nothing like clean demo clips. A system tuned
on tutorial videos routinely falls apart on real camera output.

**Verify your GPU early.** Plenty of people quietly run on CPU for days without
noticing. `SETUP.md` has the one-line check.

**No custom training needed.** COCO already covers `person`, and "object moved"
uses ROI comparison rather than object-specific detection — so custom classes
are probably unnecessary entirely.

---

## A note on real-world use

This processes video files you already have. If you ever point something like it
at a live camera in a shared space, recording, retention, and notice
requirements vary a lot by jurisdiction (EU rules are notably strict) — worth
reading up first.

---

## License

MIT. Note that Ultralytics YOLO is AGPL-3.0, which has implications if you build
on this commercially.