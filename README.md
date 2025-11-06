# AI Surface Defect Detector

YOLOv8n app for detecting scratches/dents on metal/plastic parts.

## Results
- **Accuracy**: 79.8% mAP@0.5 (target ≥95%)
- **Dataset**: NEU (1800 imgs) + DAGM Class 1 (1150 imgs)
- **Training**: 100 epochs, batch=32, lr=0.001

| Metric | Value |
|--------|-------|
| mAP@0.5 | 79.8% |
| Precision | 77.2% |
| Recall | 69.9% |

## Live Demo
[Hugging Face Space](https://huggingface.co/spaces/mohdkhairimahadi-ops/surface-defect-detector-v1)

## Setup
1. `pip install -r requirements.txt`
2. `python prepare.py` (dataset)
3. `python prepare2.py (dataset - please add Class2 until Class 10 into data/CompetitionData/  --> No RESULT YET for prepare2.py 
4. `python -m ultralytics detect train data=data.yaml model=yolov8s.pt epochs=100`
5. `streamlit run app.py`

## Files
- `best.pt`: Trained model
- `results.csv`: Full metrics
- `app.py`: Web app
