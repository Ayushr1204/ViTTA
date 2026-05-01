"""Quick script to display training progress."""
import csv

rows = []
with open(r'runs\detect\runs\train\vitta_merged\results.csv') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# Keep last occurrence of each epoch
seen = {}
for r in rows:
    seen[int(r['epoch'])] = r

print(f"{'Epoch':>5} | {'mAP50':>7} | {'mAP50-95':>8} | {'Prec':>6} | {'Recall':>6} | {'box_loss':>8} | {'cls_loss':>8}")
print("-" * 70)
for ep in sorted(seen.keys()):
    r = seen[ep]
    m50 = float(r["metrics/mAP50(B)"])
    m95 = float(r["metrics/mAP50-95(B)"])
    prec = float(r["metrics/precision(B)"])
    rec = float(r["metrics/recall(B)"])
    bl = float(r["train/box_loss"])
    cl = float(r["train/cls_loss"])
    print(f"{ep:>5} | {m50:>7.3f} | {m95:>8.3f} | {prec:>6.3f} | {rec:>6.3f} | {bl:>8.4f} | {cl:>8.4f}")

latest = max(seen.keys())
t = float(seen[latest]["time"])
avg_per_epoch = t / latest / 60
remaining = (50 - latest) * avg_per_epoch
print()
print(f"Current: epoch {latest} / 50")
print(f"Avg time per epoch: ~{avg_per_epoch:.0f} minutes")
print(f"Estimated remaining: ~{remaining:.0f} minutes ({remaining/60:.1f} hours)")
