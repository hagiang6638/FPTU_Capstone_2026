# VieSL Unified Demo App

This is the cleaned local demo for **Deep Learning Approaches for Sign Language Understanding**.
It supports:

- ISLR: isolated Vietnamese sign recognition.
- CSLR: continuous Vietnamese sign recognition.
- Upload video inference.
- Camera-based near real-time segmentation. The app starts recognition when hands are detected and closes the segment after hands disappear.
- Model selection per task through `config/models.json`.

## Folder Layout

```text
app_demo/
  app.py
  config/models.json
  src/
    inference.py
    models.py
    skeleton.py
  outputs/
  requirements.txt
```


## Run Locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Model Registry

Edit `config/models.json` when a new checkpoint is ready. Each model entry must define:

- `task`: `islr` or `cslr`
- `architecture`: supported architecture name
- `checkpoint`: model checkpoint path
- `label_map` for ISLR or `vocab` for CSLR
- `config_path`: training config saved with the run

Current supported architectures:

- `lite_tcn_bigru_islr`
- `pipeline_islr`
- `mska_plus_islr`
- `lstm_attention_islr`
- `pipeline_cslr`
- `mska_plus_cslr`
- `lite_tcn_bigru_cslr`
- `lstm_attention_cslr`

