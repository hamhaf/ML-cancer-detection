# ML-cancer-detection
In the label set, 1 means squamous tissue, 2 means non-dysplastic Barrett's oesophagus, and 3 means neoplasia (cancerous tissue)

# Instructions to run

1. clone the repository from https://github.com/hamhaf/ML-cancer-detection

2. create a venv 
    "python -m venv .venv"

3. pip install the requirements file 
    "pip install -r requirements.txt"

4. to obtain the final model, run models/models.ipynb with only FEATURE_SELECT = True and AUGMENTEDv3 = True in code block [2]

5. observe metrics in "metrics/augmentedv3_dataset/model_metrics_accuracy_ensemble_FS.json"