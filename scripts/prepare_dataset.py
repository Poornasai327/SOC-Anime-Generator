
"""
prepare_dataset_hair_balanced.py

Creates a 30k metadata file by balancing HAIR classes while preserving
the original eye-color proportions within each hair class.
"""

from collections import defaultdict
from pathlib import Path
import random
import pandas as pd
from tqdm import tqdm

INPUT = Path("dataset/all_data.csv")
OUTPUT = Path("dataset/selected_dataset.csv")

TARGET_TOTAL = 30000
HAIR_TARGET = TARGET_TOTAL // 10
SEED = 42

HAIR = {
    "black_hair","brown_hair","blonde_hair","blue_hair","pink_hair",
    "red_hair","green_hair","purple_hair","white_hair","silver_hair"
}
EYES = {
    "blue_eyes","brown_eyes","green_eyes",
    "red_eyes","yellow_eyes","purple_eyes"
}

random.seed(SEED)

def one(tags, valid):
    x=[t for t in tags if t in valid]
    return x[0] if len(x)==1 else None

print("Loading...")
df=pd.read_csv(INPUT, low_memory=False)
df=df[df["rating"]=="s"]

hair_groups=defaultdict(list)

for _,row in tqdm(df.iterrows(), total=len(df), desc="Filtering"):
    tags=str(row["tags"]).split()
    h=one(tags,HAIR)
    e=one(tags,EYES)
    if h and e:
        url = str(row["sample_url"]).strip()
        if url.startswith("//"):
            url = "https:" + url
        hair_groups[h].append({
            "id":row["id"],
            "sample_url":url,
            "hair":h,
            "eyes":e
        })

selected=[]

for hair,items in hair_groups.items():
    eye_groups=defaultdict(list)
    for it in items:
        eye_groups[it["eyes"]].append(it)

    total=len(items)
    quotas={}
    used=0
    remainders=[]

    for eye,arr in eye_groups.items():
        q=(len(arr)/total)*HAIR_TARGET
        base=min(len(arr), int(q))
        quotas[eye]=base
        used+=base
        remainders.append((q-base, eye))

    remaining=HAIR_TARGET-used
    remainders.sort(reverse=True)

    idx=0
    while remaining>0 and remainders:
        _,eye=remainders[idx % len(remainders)]
        if quotas[eye] < len(eye_groups[eye]):
            quotas[eye]+=1
            remaining-=1
        idx+=1
        if idx>100000:
            break

    for eye,arr in eye_groups.items():
        random.shuffle(arr)
        selected.extend(arr[:quotas[eye]])

random.shuffle(selected)
selected=selected[:TARGET_TOTAL]

out=pd.DataFrame(selected)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT,index=False)

print("Saved:",len(out))
print("\nHair")
print(out["hair"].value_counts().sort_index())
print("\nCross-tab")
print(pd.crosstab(out["hair"], out["eyes"]))
