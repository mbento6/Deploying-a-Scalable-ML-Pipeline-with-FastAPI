# Model Card
Last updated: 2025-07-16
Model artifact: model/model.pkl
Encoder artifact: model/encoder.pkl
Label binarizer artifact: model/label_binarizer.pkl
Training script: train_model.py
Library versions: Python 3.10.18, scikit-learn ≥1.4, pandas ≥2.2

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a RandomForestClassifier trained to predict whether an individual earns more than $50K annually based on U.S. Census data. The model was trained using sklearn’s RandomForestClassifier with GridSearchCV to optimize hyperparameters such as number of estimators, maximum depth, and minimum samples split.

## Intended Use
The model is designed for educational purposes and to demonstrate how to build and deploy a scalable machine learning pipeline using FastAPI. It is not intended for real-world decision-making without further validation and fairness analysis.

## Training Data
The model was trained using the “census.csv” dataset, split into 80% training and 20% testing sets. The target variable is “salary” (<=50K or >50K). Categorical features such as education, occupation, and workclass were one-hot encoded. The dataset should be preprocessed using the provided `process_data` function before inference.

## Evaluation Data
To ensure fairness and understand subgroup performance, metrics were computed on slices of the data by unique values of categorical features. Example slice performance includes:

- workclass: Private (Count: 4,595)
 - Precision: 0.7500 | Recall: 0.8182 | F1: 0.7826

- workclass: Federal-gov (Count: 188)
  - Precision: 0.5000 | Recall: 0.4762 | F1: 0.4878

- workclass: Self-emp-not-inc (Count: 495)
  - Precision: 0.7500 | Recall: 0.8182 | F1: 0.7826

Full slice-level results are stored in `slice_output.txt`.

## Metrics
The model was evaluated using Precision, Recall, and F1 score. The best model achieved the following metrics on the test set:
- Precision: 0.6143
- Recall: 0.8141
- F1 Score: 0.7002

## Ethical Considerations
The model was trained on historical census data and may reflect biases inherent in the source. Categorical encoding and sampling limitations can also influence model generalization. This model does not include any bias mitigation or fairness audits and is not appropriate for high-stakes decision-making.

## Caveats and Recommendations
This model and code are provided for educational purposes under the MIT License. See LICENSE for details.

# Contact Information
For issues, questions, or collaboration inquiries, contact: mbento6@wgu.edu