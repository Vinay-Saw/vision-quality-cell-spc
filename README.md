# In-line Vision Quality Cell with SPC

Portfolio prototype: one inspection station for PCB surface defects.

## Dataset
- **DeepPCB** (Tang et al., 2019): 1,500 image pairs of bare PCB, 6 defect types (open, short, mousebite, spur, pin-hole, spurious copper).
- License: **MIT** (https://github.com/tangsanli5201/DeepPCB).
- Demo uses a subset of the tested images + annotations. Images stored by SHA-256 content hash.

## What this demonstrates
- Pass / Fail / **Unsure** (first-class). Unsure holds for technician.
- Defect codes on decisions.
- Model version stamped on every row (`v1.0-classical-diff-20261013`). Decisions are immutable; human corrections stored separately as training evidence.
- SPC: p-chart (control limits, sample size n=20) + defect Pareto. Special-cause simulation (temp spike).
- Metrics that matter: escapes vs false rejects. Cost assumption: escape = 20× false reject.
- Process tags (cavity, temperature offset, shift) linked to each decision so a spike can be filtered.
- Exportable quality record (CSV) an auditor can read.

## Bias
Thresholds are set to prefer catching defects (reduce escapes) at the expense of some false rejects. Escape is more expensive.

## Stack
- Flask + SQLite
- Classical baseline (annotation-aware + simulated confidence for demo; real line would use fine-tuned detector or template-diff)
- Chart.js for SPC
- Images by content hash

## Run
```bash
cd vision-quality-cell
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

## Industry signal (what a real line needs)
- Trigger sensor (part present)
- Reject mechanism (air jet / diverter) for FAIL
- Controlled lighting + fixed camera angle + vibration isolation (demo uses fixed dataset; plant lighting/angle changes break models)
- Edge device or vision PC with deterministic latency
- ISA-95 / MES unit ID linkage (genealogy)
- Model registry + frozen versions; human corrections feed active learning offline
- Calibration to process tags (temperature, pressure, cavity)

## Interview line
“I did not optimize accuracy. I optimized cost of a wrong decision, and I kept a person on the unsure pile.”
