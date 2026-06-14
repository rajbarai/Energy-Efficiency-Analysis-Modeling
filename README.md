# Architectural Design & Building Energy Efficiency: A Machine Learning Benchmark Study

## 📋 Project Overview
This repository hosts an end-to-end data analytics and predictive modeling pipeline that evaluates how structural architectural parameters dictate a building's thermal efficiency[cite: 3, 4]. Utilizing a dense parametric experimental dataset, this project establishes rigorous statistical baselines and maps classification performance across eight distinct machine learning and deep learning frameworks[cite: 3, 4]. 

The primary business objective is to provide actionable, data-backed recommendations for structural engineers and architects to minimize structural energy demands, optimize HVAC load requirements, and design climate-resilient structures[cite: 3].

### 👥 Team Contributors (Group 4)
* Melissa Nare
* Amita Gadkari
* Raj Barai

---

## 🏗️ Technical Toolkit & Repository Structure
* **Language:** R
* **Core Libraries:** `car` (VIF Diagnostics), `neuralnet` & `NeuralNetTools` (Deep Learning), `randomForest`, `xgboost`, `e1071` (SVM), `class` (KNN), `naivebayes`, `rpart`, `caret`, `ggplot2`[cite: 4]

### File Architecture
```text
├── Data/
│   └── EnergyEfficiency.xlsx         # Raw dataset containing 768 structural observations
├── Scripts/
│   └── energy_efficiency_ml.R       # Production-ready master execution script
├── Output/
│   ├── scatterplot_matrix.png       # Initial exploratory correlation graphics
│   ├── neural_network_y1.png        # MLP architecture graph for Heating Load
│   └── decision_boundary_svm.png    # Spatial classification decision boundary map
└── README.md                        # Project documentation (This file)
