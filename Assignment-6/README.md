# Absenteeism Prediction System

A machine learning web application that predicts employee absenteeism hours using a Linear Regression model with bias mitigation techniques. The system includes explainability features (SHAP, LIME, Counterfactuals) and a React-based user interface.

## 🎯 Overview

This project builds on previous assignments to create a complete, deployable ML application for predicting workplace absenteeism. It features:

- **ML Model**: Linear Regression trained with bias-aware preprocessing
- **Backend API**: Flask application serving predictions and explanations
- **Frontend UI**: React-based interface for predictions and model insights
- **Explainability**: SHAP, LIME, and Counterfactual explanations
- **Fairness**: Bias evaluation and mitigation across age, education, and service time groups
- **Deployment**: Docker containerization with Render deployment configuration


## 🏗️ Project Structure

```
.
├── app.py                      # Flask backend API
├── explainability.py           # XAI endpoints (SHAP, LIME, CF)
├── saving_model.py             # Model training with bias evaluation
├── model.pkl                   # Trained model, scaler, and feature names
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Multi-stage Docker build
├── render.yaml                 # Render deployment configuration
├── frontend/                   # React frontend application
│   ├── src/                    # React components and pages
│   ├── package.json            # Node dependencies
│   ├── vite.config.js          # Vite bundler configuration
│   └── dist/                   # Built static files (generated)
├── tests/                      # Unit tests
│   └── test_explainability.py
└── scripts/                    # Utility scripts
    └── eval_explainability.py
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+ and npm
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/krishna-kumar-bais/Assignment__6_
   cd Assignment__6_
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up Frontend**
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

4. **Ensure model file exists**
   - The trained model should be at `model.pkl`
   - If missing, train the model: `python saving_model.py`

### Running Locally


Terminal 1 (Backend):
```bash
python app.py
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```



