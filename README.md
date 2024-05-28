## Overview
This project focuses on building a University Admission Prediction Model using machine learning techniques. The model predicts the probability of admission for prospective university students based on various academic and personal attributes.

## Dataset
The dataset used contains 500 entries with 8 columns, including predictors such as GRE score, TOEFL score, university rating, statement of purpose (SOP), letter of recommendation (LOR), CGPA, research experience, and the target variable "Chance_of_Admit."

## File Structure
- `admission.csv`: Dataset containing student admission information.
- `University_Admission_Prediction_Model.ipynb`: Jupyter Notebook containing Python code for data preprocessing, EDA, model building, and evaluation.
- `tree.png`: Decision tree visualization.
- `abc.dot`: Graphviz file for visualization of a decision tree.

## Steps to Reproduce
1. Ensure all dependencies are installed (`pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `plotly`, `pydot`, `StandardScaler`, `MinMaxScaler`).
2. Clone or download the repository to your local machine.
3. Open `University_Admission_Prediction_Model.ipynb` using Jupyter Notebook or any compatible environment.
4. Run each cell sequentially to execute the code and reproduce the results.
5. The notebook includes steps for data preprocessing, EDA, model building (Decision Tree Classifier and Random Forest Classifier), hyperparameter tuning, model evaluation, and comparison.
6. Refer to code comments for detailed explanations.

## Analysis Steps
- Data Preprocessing: Handling missing values, encoding categorical variables, feature scaling.
- Exploratory Data Analysis (EDA): Visualizing distributions, correlations, and relationships between variables.
- Model Building: Implementing Decision Tree Classifier and Random Forest Classifier.
- Hyperparameter Tuning: Optimizing model parameters using cross-validation.
- Model Evaluation: Assessing model performance using metrics like accuracy, precision, recall, and F1-score.

## Results
- Two models were evaluated: Decision Tree Classifier and Random Forest Classifier.
- Random Forest Classifier outperformed Decision Tree Classifier in accuracy and generalization.
- Model evaluation metrics indicated satisfactory performance for predicting admission probabilities.

## Conclusion
The project demonstrates the application of machine learning in predicting university admissions based on student attributes. These models can aid educational institutions in making informed decisions during the admissions process.

## Future Work
- Explore advanced feature engineering techniques to enhance model performance.
- Experiment with alternative machine learning algorithms and ensemble methods.
- Collect additional data to improve model robustness and generalization.
- Consider deploying the model as a web application for real-time predictions.

## Dependencies
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `plotly`
- `pydot`
- `StandardScaler`
- `MinMaxScaler`

## Usage
This project can be used as a reference for building admission prediction models or as a learning resource for data preprocessing, EDA, and model evaluation techniques.

## Credits
- This project was developed by Atharva Kate.
