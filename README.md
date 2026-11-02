
# Vehicle Silhouette Classification
## Machine Learning Assignment 2

### Project Overview
This project compares five supervised machine learning
classification algorithms using the Vehicle Silhouettes dataset.

### Dataset Information
- Total records: 846
- Input features: 18
- Target variable: class
- Vehicle classes: Bus, Opel, Saab, Van
- Training data: 80%
- Testing data: 20%

### Classification Algorithms
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

### Data Preprocessing
- Dataset validation
- Label encoding
- Train-test splitting
- StandardScaler normalization

### Evaluation Metrics
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient

### Best Model
Logistic Regression achieved 81.76% accuracy.

### Application Features
- Upload CSV test data
- Select classification models
- Generate predictions
- View evaluation metrics
- View confusion matrices
- Download predictions

### Run Application

Install dependencies:

python -m pip install -r requirements.txt

Launch application:

python -m streamlit run app.py

### Deployment
Platform: Streamlit Community Cloud

GitHub Repository: To be added

Live Application URL: To be added
