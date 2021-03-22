import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from flask import Flask, request, jsonify
from pathlib import Path

app = Flask(__name__)

fe = FeatureExtractor()


@app.route('/api/image/search', methods=['POST'])
def simple_image_search():
    file = request.files['query_img']
    tenant = request.form.get('tenant')

    feature_path = "static/" + tenant + "/features"
    # Check and make tenant folder
    Path(feature_path).mkdir(parents=True, exist_ok=True)

    features = []
    filenames = []
    for feature in Path(feature_path).glob("*.npy"):
        features.append(np.load(feature))
        filenames.append(feature.stem)
    features = np.array(features)

    # Run search
    query = fe.extract(Image.open(file.stream))
    # L2 distances to features
    dists = np.linalg.norm(features - query, axis=1)
    ids = np.argsort(dists)[:5]  # Top 5 results
    scores_tenant = [
        {"item_id": str(filenames[id]), "dist": str(dists[id])} for id in ids]

    # Save feature
    feature_path = feature_path + "/" + \
                   (file.filename.split('.')[0] + ".npy")
    np.save(feature_path, query)

    return jsonify(scores_tenant)


if __name__ == "__main__":
    app.run("0.0.0.0", port=9100)
