from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/HI-Small_Trans.csv")
OUT_PATH = Path("data/sample/demo_upload.csv")

CHUNKSIZE = 200_000
NORMAL_LIMIT = 50_000
LAUNDERING_LIMIT = 2_000

normal_parts = []
laundering_parts = []

for chunk in pd.read_csv(RAW_PATH, chunksize=CHUNKSIZE):
    label_col = "Is Laundering"

    laundering = chunk[chunk[label_col] == 1]
    normal = chunk[chunk[label_col] == 0]

    if len(laundering_parts) < LAUNDERING_LIMIT:
        laundering_parts.append(laundering)

    current_normal = sum(len(x) for x in normal_parts)
    if current_normal < NORMAL_LIMIT:
        normal_parts.append(normal.head(NORMAL_LIMIT - current_normal))

    current_laundering = sum(len(x) for x in laundering_parts)
    current_normal = sum(len(x) for x in normal_parts)

    if current_normal >= NORMAL_LIMIT and current_laundering >= LAUNDERING_LIMIT:
        break

demo = pd.concat(normal_parts + laundering_parts, ignore_index=True)
demo = demo.sample(frac=1, random_state=42).reset_index(drop=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
demo.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("Rows:", len(demo))
print(demo["Is Laundering"].value_counts())
print("Size MB:", OUT_PATH.stat().st_size / 1024 / 1024)
