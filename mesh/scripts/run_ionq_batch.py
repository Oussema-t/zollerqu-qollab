"""Local runner for the SAME single IonQ batch through the IonQ cloud (needs `pip install qiskit-ionq` and an IonQ API key
in the environment variable IONQ_API_KEY - never in a file). Uses the Forte 1 noise model and saves the record to
qollab/results_ionq/, which Qollab's main.py then shows as the recorded run.
usage: IONQ_API_KEY=... python scripts/run_ionq_batch.py [shots]"""
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import qiskit
from qiskit_ionq import IonQProvider

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "qollab"))
import mesh_ionq as mi  # noqa: E402

shots = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
batch = json.loads((ROOT / "qollab" / "mesh_batch.json").read_text())
for scenario in batch["scenarios"]:
    mi.check_exact_optimum(scenario, batch["n"])
backend = IonQProvider(os.environ["IONQ_API_KEY"]).get_backend("simulator")
circuits, labels = mi.build_batch(batch)
compiled = qiskit.transpile(circuits, backend, optimization_level=1)
job = backend.run(compiled, shots=shots, noise_model="forte-1")
print("submitted job", job.job_id())
result = job.result()
counts = [result.get_counts(i) for i in range(len(compiled))]
scenario_of = {s["name"]: s for s in batch["scenarios"]}
metrics = [mi.grade(c, scenario_of[label[0]], batch["n"]) for c, label in zip(counts, labels)]
record = dict(backend="IonQ Forte 1 noise model (IonQ cloud simulator)", date=str(date.today()), timestamp=datetime.now(timezone.utc).isoformat(),
              shots=shots, job_id=job.job_id(), qiskit_version=qiskit.__version__, labels=labels, metrics=metrics, counts=counts,
              predicted=[[c["predicted"] for c in s["circuits"]] for s in batch["scenarios"]])
out = ROOT / "qollab" / "results_ionq" / f"mesh_forte1_{batch['n']}q_{date.today()}.json"
out.write_text(json.dumps(record))
print("saved", out)
