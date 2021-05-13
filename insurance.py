import glob
import os

import numpy as np
import pandas as pd
from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path

fe = FeatureExtractor()

base_dir = "modified"
feature_path = "static/feature"

features = []
filenames = []
for feature in Path(feature_path).glob("*.npy"):
    features.append(np.load(feature))
    filenames.append(feature.stem)
features = np.array(features)

total_results = []
for filename in glob.glob(f"static/{base_dir}/*.jpg"):
    query = fe.extract(Image.open(filename))
    dists = np.linalg.norm(features - query, axis=1)
    ids = np.argsort(dists)[:2]
    results = [[filenames[id], dists[id]] for id in ids]
    results = [os.path.basename(filename)] + np.array(results).flatten().tolist()
    total_results.append(results)

df = pd.DataFrame(np.array(total_results),
                  columns=["filename", "nearest", "distance", "second_nearest", "distance"])
df.to_csv("result.csv", index=False)
