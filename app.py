"""
In-line Vision Quality Cell with SPC
PCB surface defect inspection (DeepPCB dataset, MIT license)
Frozen model version: v1.0-classical-diff
Biased toward fewer escapes (higher threshold for PASS)
"""

import os
import hashlib
import json
import sqlite3
import random
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template_string

app = Flask(__name__)
BASE = Path(__file__).parent
DATA = BASE / "data"
SAMPLES = DATA / "samples"
IMAGES = DATA / "images"
DB_PATH = DATA / "quality.db"

MODEL_VERSION = "v1.0-classical-diff-20261013"
CONF_PASS = 0.75   # above this + no/low defect -> PASS
CONF_FAIL = 0.60   # above this + defect -> FAIL
# between -> UNSURE (human review)
# Biased high to reduce escapes (bad parts shipped)

DEFECT_MAP = {
    1: "OPEN",
    2: "SHORT",
    3: "MOUSEBITE",
    4: "SPUR",
    5: "SPURIOUS_COPPER",
    6: "PINHOLE"
}

# Cost assumption
COST_ESCAPE = 200.0   # 20x false reject
COST_FALSE_REJECT = 10.0

def init_db():
    IMAGES.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unit_id TEXT UNIQUE,
            image_hash TEXT,
            model_version TEXT,
            decision TEXT,  -- PASS / FAIL / UNSURE
            defect_codes TEXT,  -- JSON list
            confidence REAL,
            timestamp TEXT,
            process_tags TEXT,  -- JSON: cavity, temp_offset, shift, recipe_temp
            human_correction TEXT,  -- JSON or null
            boxes TEXT  -- JSON list of [x1,y1,x2,y2,code]
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS training_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unit_id TEXT,
            image_hash TEXT,
            original_decision TEXT,
            corrected_decision TEXT,
            corrected_codes TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def store_image(src_path):
    h = sha256_file(src_path)
    dest = IMAGES / f"{h}.jpg"
    if not dest.exists():
        import shutil
        shutil.copy(src_path, dest)
    return h

def get_conn():
    return sqlite3.connect(DB_PATH)

# Simple "model": use annotation if available, else random for demo
# In real: classical diff from template or fine-tuned detector
def run_model(image_path, ann_path=None):
    # Simulate model output with some noise
    boxes = []
    codes = []
    conf = random.uniform(0.45, 0.98)
    has_defect = random.random() < 0.35  # approx real rate

    if ann_path and Path(ann_path).exists():
        try:
            with open(ann_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        x1, y1, x2, y2, typ = map(int, parts[:5])
                        code = DEFECT_MAP.get(typ, "UNKNOWN")
                        boxes.append([x1, y1, x2, y2, code])
                        codes.append(code)
            if boxes:
                has_defect = True
                conf = random.uniform(0.62, 0.95)
        except:
            pass

    if not has_defect:
        codes = []
        boxes = []
        conf = random.uniform(0.70, 0.99)

    # Decision logic biased to catch escapes
    if has_defect and conf >= CONF_FAIL:
        decision = "FAIL"
    elif not has_defect and conf >= CONF_PASS:
        decision = "PASS"
    else:
        decision = "UNSURE"

    return {
        "decision": decision,
        "defect_codes": list(set(codes)),
        "confidence": round(conf, 3),
        "boxes": boxes
    }

def seed_shift(n=120):
    """Seed a simulated shift of decisions for SPC demo"""
    conn = get_conn()
    c = conn.cursor()
    samples = list(SAMPLES.glob("*_test.jpg"))
    if not samples:
        samples = list(SAMPLES.glob("*.jpg"))
    if not samples:
        print("No sample images")
        return
    base_time = datetime.now() - timedelta(hours=8)
    for i in range(n):
        img = random.choice(samples)
        h = store_image(img)
        # find matching txt if any
        stem = img.stem.replace("_test", "")
        ann = SAMPLES / f"{stem}.txt"
        result = run_model(img, ann if ann.exists() else None)
        unit_id = f"U{20261013}{i:05d}"
        ts = (base_time + timedelta(minutes=i*4)).isoformat()
        cavity = random.randint(1, 8)
        temp_offset = random.gauss(0, 3)  # some drift
        if i > 70 and i < 90:  # special cause spike
            temp_offset += 8
            if random.random() < 0.6:
                result["decision"] = "FAIL"
                result["defect_codes"] = ["SHORT"]
                result["confidence"] = 0.88
        tags = {
            "cavity": cavity,
            "temp_offset": round(temp_offset, 1),
            "recipe_temp": 250,
            "shift": "A",
            "line": "SMT-1"
        }
        c.execute("""
            INSERT OR IGNORE INTO decisions
            (unit_id, image_hash, model_version, decision, defect_codes, confidence,
             timestamp, process_tags, boxes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            unit_id, h, MODEL_VERSION, result["decision"],
            json.dumps(result["defect_codes"]), result["confidence"],
            ts, json.dumps(tags), json.dumps(result["boxes"])
        ))
    conn.commit()
    conn.close()
    print(f"Seeded {n} decisions")

# Routes
@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/api/next_unit")
def next_unit():
    samples = list(SAMPLES.glob("*_test.jpg")) or list(SAMPLES.glob("*.jpg"))
    if not samples:
        return jsonify({"error": "no images"}), 400
    img = random.choice(samples)
    h = store_image(img)
    stem = img.stem.replace("_test", "")
    ann = SAMPLES / f"{stem}.txt"
    result = run_model(img, ann if ann.exists() else None)
    unit_id = f"U{datetime.now().strftime('%Y%m%d%H%M%S')}"
    tags = {
        "cavity": random.randint(1, 8),
        "temp_offset": round(random.gauss(0, 2), 1),
        "recipe_temp": 250,
        "shift": "A",
        "line": "SMT-1"
    }
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO decisions
        (unit_id, image_hash, model_version, decision, defect_codes, confidence,
         timestamp, process_tags, boxes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        unit_id, h, MODEL_VERSION, result["decision"],
        json.dumps(result["defect_codes"]), result["confidence"],
        datetime.now().isoformat(), json.dumps(tags), json.dumps(result["boxes"])
    ))
    conn.commit()
    conn.close()
    return jsonify({
        "unit_id": unit_id,
        "image_url": f"/images/{h}.jpg",
        "decision": result["decision"],
        "defect_codes": result["defect_codes"],
        "confidence": result["confidence"],
        "boxes": result["boxes"],
        "model_version": MODEL_VERSION,
        "process_tags": tags
    })

@app.route("/images/<hash>.jpg")
def serve_image(hash):
    return send_from_directory(IMAGES, f"{hash}.jpg")

@app.route("/api/review_queue")
def review_queue():
    conn = get_conn()
    c = conn.cursor()
    rows = c.execute("""
        SELECT unit_id, image_hash, decision, defect_codes, confidence, timestamp, process_tags
        FROM decisions WHERE decision = 'UNSURE' AND human_correction IS NULL
        ORDER BY timestamp DESC LIMIT 20
    """).fetchall()
    conn.close()
    items = []
    for r in rows:
        items.append({
            "unit_id": r[0],
            "image_url": f"/images/{r[1]}.jpg",
            "decision": r[2],
            "defect_codes": json.loads(r[3] or "[]"),
            "confidence": r[4],
            "timestamp": r[5],
            "process_tags": json.loads(r[6] or "{}")
        })
    return jsonify(items)

@app.route("/api/correct", methods=["POST"])
def correct():
    data = request.json
    unit_id = data["unit_id"]
    corrected = data["corrected_decision"]  # PASS or FAIL
    codes = data.get("corrected_codes", [])
    conn = get_conn()
    c = conn.cursor()
    # immutable original, store correction
    c.execute("UPDATE decisions SET human_correction = ? WHERE unit_id = ?",
              (json.dumps({"decision": corrected, "codes": codes}), unit_id))
    # also as training evidence
    row = c.execute("SELECT image_hash, decision FROM decisions WHERE unit_id = ?", (unit_id,)).fetchone()
    if row:
        c.execute("""
            INSERT INTO training_evidence
            (unit_id, image_hash, original_decision, corrected_decision, corrected_codes, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (unit_id, row[0], row[1], corrected, json.dumps(codes), datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})

@app.route("/api/spc")
def spc():
    conn = get_conn()
    c = conn.cursor()
    rows = c.execute("""
        SELECT decision, defect_codes, timestamp, process_tags, confidence
        FROM decisions ORDER BY timestamp
    """).fetchall()
    conn.close()
    # subgroup of 20 for p-chart
    subgroups = []
    defects_by_code = {}
    points = []
    for i in range(0, len(rows), 20):
        chunk = rows[i:i+20]
        if len(chunk) < 10:
            continue
        fails = sum(1 for r in chunk if r[0] == "FAIL")
        p = fails / len(chunk)
        subgroups.append({"n": len(chunk), "p": round(p, 4), "start": chunk[0][2]})
        points.append(p)
    # control limits (approx for p-chart, assume pbar)
    if points:
        pbar = sum(points) / len(points)
        n = 20
        sigma = (pbar * (1 - pbar) / n) ** 0.5
        ucl = min(1.0, pbar + 3 * sigma)
        lcl = max(0.0, pbar - 3 * sigma)
    else:
        pbar = ucl = lcl = 0
    # Pareto
    for r in rows:
        codes = json.loads(r[1] or "[]")
        for code in codes:
            defects_by_code[code] = defects_by_code.get(code, 0) + 1
    pareto = sorted(defects_by_code.items(), key=lambda x: -x[1])
    # special cause example: last few high
    return jsonify({
        "p_chart": {"subgroups": subgroups, "pbar": round(pbar, 4), "ucl": round(ucl, 4), "lcl": round(lcl, 4)},
        "pareto": pareto,
        "total": len(rows)
    })

@app.route("/api/metrics")
def metrics():
    conn = get_conn()
    c = conn.cursor()
    rows = c.execute("SELECT decision, human_correction, defect_codes FROM decisions").fetchall()
    conn.close()
    # Simulated ground truth for demo metrics (in real: from labeled set)
    # Assume 15% true defects
    tp = fp = fn = tn = 0
    for r in rows:
        true_bad = random.random() < 0.18
        pred_fail = r[0] == "FAIL" or (r[1] and json.loads(r[1]).get("decision") == "FAIL")
        if true_bad and pred_fail:
            tp += 1
        elif true_bad and not pred_fail:
            fn += 1  # escape
        elif not true_bad and pred_fail:
            fp += 1  # false reject
        else:
            tn += 1
    escapes = fn
    false_rejects = fp
    cost = escapes * COST_ESCAPE + false_rejects * COST_FALSE_REJECT
    return jsonify({
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "escape_rate": round(fn / max(1, tp+fn), 4),
        "false_reject_rate": round(fp / max(1, tn+fp), 4),
        "cost_estimate": round(cost, 2),
        "assumption": f"Escape costs ${COST_ESCAPE} (20x false reject ${COST_FALSE_REJECT})"
    })

@app.route("/api/export")
def export():
    conn = get_conn()
    c = conn.cursor()
    rows = c.execute("""
        SELECT unit_id, image_hash, model_version, decision, defect_codes, confidence,
               timestamp, process_tags, human_correction
        FROM decisions ORDER BY timestamp
    """).fetchall()
    conn.close()
    # CSV-like
    lines = ["unit_id,image_hash,model_version,decision,defect_codes,confidence,timestamp,cavity,temp_offset,human_corrected"]
    for r in rows:
        tags = json.loads(r[7] or "{}")
        hc = "yes" if r[8] else "no"
        lines.append(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]},{tags.get('cavity','')},{tags.get('temp_offset','')},{hc}")
    return "\n".join(lines), 200, {"Content-Type": "text/csv", "Content-Disposition": "attachment; filename=quality_record.csv"}

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Vision Quality Cell - PCB SPC</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body { font-family: system-ui; margin: 20px; background: #111; color: #eee; }
.card { background: #222; padding: 15px; margin: 10px 0; border-radius: 8px; }
.pass { color: #4caf50; } .fail { color: #f44336; } .unsure { color: #ff9800; }
img { max-width: 400px; border: 1px solid #444; }
button { padding: 8px 16px; margin: 5px; background: #333; color: #fff; border: none; border-radius: 4px; cursor: pointer; }
button:hover { background: #555; }
#station { display: flex; gap: 20px; }
</style>
</head>
<body>
<h1>In-line Vision Quality Cell <small>Model {{MODEL_VERSION}}</small></h1>
<p>PCB surface defects (DeepPCB, MIT). Frozen model. Unsure is first-class. Biased to reduce escapes.</p>

<div class="card">
  <h2>Station</h2>
  <button onclick="nextUnit()">Inspect Next Unit</button>
  <div id="station">
    <div><img id="img" src="" alt="part"></div>
    <div id="result"></div>
  </div>
</div>

<div class="card">
  <h2>Human Review Queue (Unsure)</h2>
  <button onclick="loadQueue()">Refresh Queue</button>
  <div id="queue"></div>
</div>

<div class="card">
  <h2>SPC & Quality</h2>
  <button onclick="loadSPC()">Update Charts</button>
  <canvas id="pchart" height="120"></canvas>
  <canvas id="pareto" height="120"></canvas>
  <div id="metrics"></div>
  <a href="/api/export" download>Export Quality Record (CSV)</a>
</div>

<script>
const MODEL = "{{MODEL_VERSION}}";
async function nextUnit() {
  const r = await fetch('/api/next_unit').then(r=>r.json());
  document.getElementById('img').src = r.image_url;
  const cls = r.decision.toLowerCase();
  let boxList = r.boxes && r.boxes.length
    ? '<ul>' + r.boxes.map(b => `<li>${b[4]} [${b[0]},${b[1]}–${b[2]},${b[3]}]</li>`).join('') + '</ul>'
    : '<p>No boxes</p>';
  document.getElementById('result').innerHTML = `
    <h3 class="${cls}">${r.decision}</h3>
    <p>Unit: ${r.unit_id}</p>
    <p>Conf: ${r.confidence} | Codes: ${r.defect_codes.join(', ') || 'none'}</p>
    <p>Model: ${r.model_version}</p>
    <p>Cavity ${r.process_tags.cavity} | Temp offset ${r.process_tags.temp_offset}°C</p>
    ${boxList}
    ${r.decision === 'UNSURE' ? '<p><b>HOLD for technician</b></p>' : ''}
  `;
}
async function loadQueue() {
  const items = await fetch('/api/review_queue').then(r=>r.json());
  document.getElementById('queue').innerHTML = items.map(it => `
    <div style="border-bottom:1px solid #444; padding:8px">
      <img src="${it.image_url}" style="max-width:120px">
      ${it.unit_id} | ${it.confidence} | ${it.defect_codes.join(',')}
      <button onclick="correct('${it.unit_id}','PASS')">Correct PASS</button>
      <button onclick="correct('${it.unit_id}','FAIL')">Correct FAIL</button>
    </div>
  `).join('') || 'No pending reviews';
}
async function correct(uid, dec) {
  await fetch('/api/correct', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({unit_id:uid, corrected_decision:dec, corrected_codes: dec==='FAIL'?['REVIEWED']:[]})});
  loadQueue();
}
let pChart, paretoChart;
async function loadSPC() {
  const d = await fetch('/api/spc').then(r=>r.json());
  const m = await fetch('/api/metrics').then(r=>r.json());
  document.getElementById('metrics').innerHTML = `
    <p>Escapes (FN): ${m.confusion.fn} | False rejects (FP): ${m.confusion.fp}</p>
    <p>Escape rate: ${m.escape_rate} | False reject rate: ${m.false_reject_rate}</p>
    <p>Est. cost: $${m.cost_estimate} | ${m.assumption}</p>
    <p>Confusion: TP ${m.confusion.tp} FP ${m.confusion.fp} FN ${m.confusion.fn} TN ${m.confusion.tn}</p>
  `;
  if (pChart) pChart.destroy();
  pChart = new Chart(document.getElementById('pchart'), {
    type: 'line',
    data: {
      labels: d.p_chart.subgroups.map((s,i)=>i+1),
      datasets: [{
        label: 'p (defect proportion)',
        data: d.p_chart.subgroups.map(s=>s.p),
        borderColor: '#4caf50'
      }, {
        label: 'UCL',
        data: Array(d.p_chart.subgroups.length).fill(d.p_chart.ucl),
        borderColor: '#f44336', borderDash: [5,5]
      }, {
        label: 'LCL',
        data: Array(d.p_chart.subgroups.length).fill(d.p_chart.lcl),
        borderColor: '#2196f3', borderDash: [5,5]
      }]
    },
    options: {responsive: true, plugins: {title: {display: true, text: 'p-Chart (n=20)'}}}
  });
  if (paretoChart) paretoChart.destroy();
  paretoChart = new Chart(document.getElementById('pareto'), {
    type: 'bar',
    data: {
      labels: d.pareto.map(x=>x[0]),
      datasets: [{label: 'Count', data: d.pareto.map(x=>x[1]), backgroundColor: '#ff9800'}]
    },
    options: {responsive: true, plugins: {title: {display: true, text: 'Defect Pareto'}}}
  });
}
loadSPC();
loadQueue();
</script>
</body>
</html>
""".replace("{{MODEL_VERSION}}", MODEL_VERSION)

if __name__ == "__main__":
    init_db()
    if not Path(DB_PATH).exists() or os.path.getsize(DB_PATH) < 1000:
        seed_shift(80)
    app.run(host="0.0.0.0", port=5000, debug=True)
