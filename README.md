# ml-model-comparison

Crime Arrest Prediction – Machine Learning Model Comparison
This project builds and evaluates multiple machine learning regression models to predict total arrests using historical crime data. The goal is to compare model performance and identify the best-performing approach based on standard regression metrics.
The pipeline includes feature correlation analysis, feature selection, model training, evaluation, and result export.
Features
Trains and evaluates 12 regression models:
Linear Regression, Ridge, Lasso, ElasticNet
Decision Tree, Random Forest, Extra Trees
Gradient Boosting, AdaBoost
K-Nearest Neighbors
Support Vector Regression
Neural Network (MLP)
Evaluation metrics: MAPE, RMSE, MAE, R²
Automatic best-model selection based on MAPE
Feature importance extraction for tree-based models
Correlation-based feature selection
CSV export of results and feature importance
Project Structure
.
├── crime_data.csv
├── model_selector.py
├── model_comparison_results.csv
├── feature_importance_<model_name>.csv
└── README.md
Dataset Requirements
The input dataset (crime_data.csv) must:
Be in CSV format
Contain numeric crime-related features
Include the following required columns:
year
total_arrests
The dataset is split as follows:
Training set: years ≤ 2012
Test set: years > 2012
Installation
Install required dependencies:
pip install pandas numpy scikit-learn
Usage
Run the script from the project directory:
python model_selector.py
The script performs the following steps:
Loads and analyzes the crime dataset
Computes feature correlations with total arrests
Selects the top correlated features
Trains and evaluates 12 machine learning models
Selects the best-performing model
Saves evaluation results and feature importance to CSV files
Model Evaluation
Models are evaluated using:
Mean Absolute Percentage Error (MAPE)
Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
R² Score
The model with the lowest MAPE is selected as the best model.
Output Files
After execution, the following files are generated:
model_comparison_results.csv
Performance metrics for all evaluated models
feature_importance_<model_name>.csv
Feature importance rankings for supported models
Feature Selection
The project uses correlation-based feature selection by default, selecting the top 15 features most correlated with the target variable. This behavior can be modified in the select_best_features function.
Customization
Potential extensions include:
Enabling cross-validation
Modifying model hyperparameters
Using alternative feature selection methods
Adding additional regression models
License
This project is intended for educational and research purposes.
