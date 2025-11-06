# AI Surface Defect Detector

YOLOv8n app for detecting scratches/dents on metal/plastic parts.

## Results
- **Accuracy**: 96.2% mAP@0.5 (target ≥95%)
- **Dataset**: NEU (1800 imgs) + DAGM Class 1 (1150 imgs)
- **Training**: 50 epochs, batch=16, lr=0.001

| Metric | Value |
|--------|-------|
| mAP@0.5 | 96.2% |
| Precision | 95.1% |
| Recall | 92.3% |

## Live Demo
[Hugging Face Space](https://huggingface.co/spaces/mohdkhairimahadi-ops/surface-defect-detector-v1)

## Setup
1. `pip install -r requirements.txt`
2. `python prepare.py` (dataset)
3. `python -m ultralytics detect train data=data.yaml model=yolov8n.pt epochs=50`
4. `streamlit run app.py`

## Files
- `best.pt`: Trained model
- `results.csv`: Full metrics
- `app.py`: Web app