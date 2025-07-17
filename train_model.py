import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

project_path = "." 
data_path = os.path.join(project_path, "data", "census.csv")
print(f"Loading data from: {data_path}")
data = pd.read_csv(data_path)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    X=train,
    categorical_features=cat_features,
    label="salary",
    training=True
    )

X_test, y_test, _, _ = process_data(
    X=test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)


model = train_model(X_train, y_train)

# save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
print(f"Model saved to {model_path}")

encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)
print(f"Encoder saved to {encoder_path}")

lb_path = os.path.join(project_path, "model", "label_binarizer.pkl")
save_model(lb, lb_path)
print(f"Label Binarizer saved to {lb_path}")

# load the model
model = load_model(model_path) 
print(f"Loading model from {model_path}")

preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test.ravel(), preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# iterate through the categorical features
output_path = os.path.join(project_path, "slice_output.txt")
with open(output_path, "w") as f:   
    for col in cat_features:
    # iterate through the unique values in one categorical feature
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]
            p, r, fb = performance_on_categorical_slice(
                data=test,
                column_name=col,
                slice_value=slicevalue,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                model=model
            )
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)

print(f"Slice metrics written to: {output_path}")
