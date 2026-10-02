"""Fill README.md with the numbers from your Kaggle run.

Usage (from the repo root, after unzipping loomguard_artifacts.zip there):
    python tools/fill_readme.py
"""
import json
import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent


def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- {name}:START -->).*?(<!-- {name}:END -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"README is missing the {name} markers.")
    return pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(2)}", text)


def main():
    res_path, table_path = ROOT / "results.json", ROOT / "RESULTS.md"   # RESULTS.md is regenerated
    if not res_path.exists():
        raise SystemExit("results.json not found. Unzip the Kaggle artifacts into the repo root first.")
    summary = json.loads(res_path.read_text())
    results = summary["results"]
    deploy = summary.get("deploy_category", "carpet")
    if deploy not in results:
        deploy = next(iter(results))

    def avg(model, key):
        return mean(results[c][model][key] for c in results)

    env = summary.get("environment", {})
    sp_cpu, pc_cpu = avg("stfpm", "cpu_latency_ms"), avg("patchcore", "cpu_latency_ms")
    size = results[deploy]["stfpm"]["onnx_size_mb"]

    rows = ["| Category | Model | Image AUROC | Pixel AUROC | F1 (label-free threshold) | False alarms | GPU ms | CPU ms |",
            "|---|---|---|---|---|---|---|---|"]
    for c in results:
        for name, label in [("stfpm", "STFPM (deployed, ONNX)"), ("patchcore", "PatchCore")]:
            m = results[c][name]
            rows.append(f"| {c} | {label} | {m['image_auroc']:.3f} | {m['pixel_auroc']:.3f} | {m['f1']:.3f} | "
                        f"{m['false_alarm_rate']:.1%} | {m['gpu_latency_ms']:.1f} | {m['cpu_latency_ms']:.1f} |")
    for name, label in [("stfpm", "STFPM (deployed, ONNX)"), ("patchcore", "PatchCore")]:
        rows.append(f"| **mean** | {label} | {avg(name, 'image_auroc'):.3f} | {avg(name, 'pixel_auroc'):.3f} | "
                    f"{avg(name, 'f1'):.3f} | {avg(name, 'false_alarm_rate'):.1%} | {avg(name, 'gpu_latency_ms'):.1f} | "
                    f"{avg(name, 'cpu_latency_ms'):.1f} |")
    table_path.write_text("\n".join(rows) + "\n")

    good = [c for c in results if results[c]["stfpm"]["f1"] >= 0.8 and results[c]["stfpm"]["false_alarm_rate"] <= 0.1]
    weak = [c for c in results if c not in good]
    min_f1 = min(results[c]["stfpm"]["f1"] for c in good) if good else 0
    weak_note = ""
    if weak:
        w = weak[0]
        weak_note = (f"\n\n**Where STFPM falls short.** On {', '.join(weak)}, STFPM still localises defects well "
                     f"(pixel AUROC {results[w]['stfpm']['pixel_auroc']:.3f}) but its image-level scores separate good and "
                     f"defective images poorly (image AUROC {results[w]['stfpm']['image_auroc']:.3f}), so the label-free "
                     f"threshold misses most defects. PatchCore reaches {results[w]['patchcore']['image_auroc']:.3f} there, "
                     f"at ~{pc_cpu / sp_cpu:.0f}x the CPU cost. So the app routes {', '.join(weak)} to a PatchCore ONNX model "
                     f"(`notebooks/loomguard_carpet_patchcore.ipynb`, 1% coreset, calibrated and evaluated through the "
                     f"exported ONNX itself) and keeps the fast STFPM model for the other textures.")

    results_md = (
        "\n".join(rows)
        + f"\n\nThresholds use no defect labels ({summary['threshold_rule']}). Image score: "
        f"{summary.get('image_score', 'max pixel')}. GPU: {env.get('gpu') or 'n/a'}. CPU timings: Kaggle CPU, "
        f"{env.get('cpu_threads')} threads, batch size 1, STFPM through ONNX Runtime and PatchCore in PyTorch."
        + weak_note
        + f"\n\n![Heatmaps on {deploy}](figures/{deploy}_heatmaps.png)\n\n"
        f"![Score distributions and ROC on {deploy}](figures/{deploy}_scores.png)"
    )
    resume_md = "\n".join([
        f"- Built **LoomGuard**, an unsupervised surface-defect detector trained only on defect-free images; "
        f"**{avg('stfpm', 'pixel_auroc'):.1%}** pixel AUROC and **{avg('stfpm', 'image_auroc'):.1%}** image AUROC across "
        f"{len(results)} MVTec AD texture categories (fabric, leather, wood, tile, mesh).",
        f"- Benchmarked a student–teacher model (STFPM) against PatchCore; deployed STFPM as a **{size} MB ONNX** model "
        f"at **{sp_cpu:.0f} ms/image on CPU**, **{pc_cpu / sp_cpu:.0f}× faster** than PatchCore.",
        f"- Set decision thresholds without any defect labels: **0.{int(min_f1 * 100):02d}–"
        f"{max(results[c]['stfpm']['f1'] for c in good):.2f} F1** with "
        f"**{max(results[c]['stfpm']['false_alarm_rate'] for c in good):.0%} or fewer false alarms** on {len(good)} of "
        f"{len(results)} textures; routed the remaining texture to PatchCore (per-category model routing) and diagnosed "
        f"threshold drift and a training divergence along the way.",
        "- Served through FastAPI + ONNX Runtime with a live camera heatmap dashboard; containerised with Docker.",
    ])
    readme = ROOT / "README.md"
    text = replace_block(readme.read_text(), "RESULTS", results_md)
    text = replace_block(text, "RESUME", resume_md)
    readme.write_text(text)
    print("README updated.\n\nResume bullets:\n" + resume_md.replace("**", ""))
    broken = [c for c in results if results[c]["stfpm"]["pixel_auroc"] < 0.9]
    print("\nHEALTH CHECK:", "training OK in all categories." if not broken else f"training failed on {broken}.")


if __name__ == "__main__":
    main()
