import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("loan_approval_dataset.csv")

# Show all column names
print(df.columns)

# Remove loan_id column if present
if 'loan_id' in df.columns:
    df = df.drop('loan_id', axis=1)

# Convert text columns into numbers
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# CHANGE THIS TARGET COLUMN NAME IF NEEDED
X = df.drop(' loan_status', axis=1)
y = df[' loan_status']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create pipeline
pipeline = Pipeline([
    ('model', KNeighborsClassifier(n_neighbors=3))
])

# Train model
pipeline.fit(X_train, y_train)
# Train model
pipeline.fit(X_train, y_train)

# Accuracy
from sklearn.metrics import accuracy_score

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# Save model
joblib.dump(pipeline, 'loan_approval_pipeline.pkl')

# Load model
model_loaded = joblib.load('loan_approval_pipeline.pkl')

# Sample data
data = pd.DataFrame(
    [[2,1,0,500000,1000000,10,750,3000000,0,500000,400000]],
    columns=X.columns
)

# Predict
pred = model_loaded.predict(data)

print(pred)

if pred[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")
    import gradio as gr

def loan_prediction(
    dependents,
    education,
    self_employed,
    income,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets,
    commercial_assets,
    luxury_assets,
    bank_assets
):

    data = pd.DataFrame(
        [[
            dependents,
            education,
            self_employed,
            income,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets,
            commercial_assets,
            luxury_assets,
            bank_assets
        ]],
        columns=X.columns
    )

    pred = model_loaded.predict(data)

    if pred[0] == 1:
        return "Loan Approved"
    else:
        return "Loan Rejected"


interface = gr.Interface(
    fn=loan_prediction,
    inputs=[
        gr.Number(label="Dependents"),
        gr.Number(label="Education"),
        gr.Number(label="Self Employed"),
        gr.Number(label="Income"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Term"),
        gr.Number(label="CIBIL Score"),
        gr.Number(label="Residential Assets"),
        gr.Number(label="Commercial Assets"),
        gr.Number(label="Luxury Assets"),
        gr.Number(label="Bank Assets")
    ],
    outputs="text",
    title="Loan Approval Prediction"
)

interface.launch()