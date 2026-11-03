
# Vehicle Silhouette Classification
## Machine Learning Assignment 2

### Problem Statement
Problem Statement: Vehicle Silhouette Classification Using Machine Learning

Vehicle classification is an important task in intelligent transportation systems, traffic monitoring, and automated vehicle recognition. Identifying vehicle types accurately from their visual characteristics can help improve traffic analysis and transportation management.

The objective of this project is to develop and evaluate machine learning classification models that can identify vehicle types based on their geometric and silhouette-related features.

The project uses the Vehicle Silhouettes dataset, consisting of 846 observations, 18 numerical input features, and four vehicle classes: Bus, Opel, Saab, and Van.

The dataset is analyzed and preprocessed before training multiple classification algorithms, including Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Naive Bayes, and Support Vector Machine (SVM).

The performance of these models is evaluated using Accuracy, Precision, Recall, F1-score, AUC, Matthews Correlation Coefficient (MCC), and Confusion Matrix.

The project also aims to develop an interactive web application using Streamlit, allowing users to upload vehicle test data, select a trained classification model, generate predictions, view model performance reports, and download prediction results.

The primary goal is to identify the most suitable machine learning model for vehicle silhouette classification and demonstrate its practical application through an interactive web-based interfac

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
