# **🚀 Foundational MLOps Pipeline**

This project is a boilerplate for building a standard, automated Machine Learning pipeline. It moves away from "Manual Notebook" development into a "Production-Ready" system using industry-standard tools.

## **🏗 System Architecture**

1. **Code & Orchestration:** GitHub & GitHub Actions.  
2. **Experiment Tracking:** MLflow (Tracking parameters, metrics, and artifacts).  
3. **Configuration Management:** params.yaml (Centralized control of model logic).  
4. **Data Versioning:** DVC (Separating large data files from Git code).  
5. **Model Serving:** FastAPI (Wrapping the model for real-world usage).

## **📂 Project Structure**

.  
├── .github/workflows/  
│   └── pipeline.yaml     \# CI/CD Automation script  
├── data/                 \# Data folder (Tracked by DVC)  
├── src/  
│   ├── train.py          \# Training logic with MLflow integration  
│   ├── evaluate.py       \# Metrics calculation  
│   └── app.py            \# FastAPI serving script  
├── params.yaml           \# Centralized hyperparameters  
├── requirements.txt      \# Python dependencies  
└── dvc.yaml              \# DVC pipeline stages

## **🛠 Prerequisites**

1. **Python 3.9+**  
2. **DVC:** pip install dvc  
3. **MLflow:** A free account on [DAGSHub](https://dagshub.com) or a local server.

## **🚀 Execution Guide**

### **1\. Setup Environment**

git init  
pip install \-r requirements.txt  
dvc init

### **2\. Configure Parameters**

Edit params.yaml to change model behavior. Every change here is tracked by Git, creating a history of your "Experiments."

### **3\. Run Training Locally**

python src/train.py

This script will:

* Load parameters from params.yaml.  
* Train a Scikit-Learn model.  
* Log the Accuracy, Precision, and the Model file itself to **MLflow**.

### **4\. Automation via GitHub Actions**

Whenever you git push a change to params.yaml or train.py, the GitHub Action in .github/workflows/pipeline.yaml will automatically:

1. Spin up a virtual Ubuntu server.  
2. Install your requirements.  
3. Run the training script.  
   Result: You can see the new model and its performance in your MLflow dashboard immediately without running anything on your laptop.

## **📈 Key Concepts to Learn**

* **Reproducibility:** If you check out an old Git commit, can you recreate the exact same model? (Yes, thanks to params.yaml and DVC).  
* **Auditability:** Who trained the model that is currently in production? (Check MLflow logs).  
* **Separation of Concerns:** Code is in Git; Data is in DVC; Models are in MLflow.