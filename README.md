# Ordinal and Label Encoding in Python

## Overview
This repository demonstrates the implementation of **Ordinal Encoding** and **Label Encoding** using Python and Scikit-Learn. The dataset used in this project is `customer.csv`, and the encoding techniques are applied to transform categorical data into numerical format for machine learning models.

## Dataset
The dataset (`customer.csv`) contains customer information, and the target variable `purchased` indicates whether a customer has made a purchase. We perform encoding on categorical features to prepare the data for modeling.

## Requirements
Ensure you have the following dependencies installed before running the code:
```bash
pip install pandas scikit-learn
```

## Implementation
### 1. Importing Libraries
```python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
```

### 2. Loading the Dataset
```python
df = pd.read_csv('/content/customer.csv')
df.sample(5)  # Display a random sample of 5 rows
```

### 3. Selecting Relevant Features
```python
df = df.iloc[:,2:]  # Dropping unnecessary columns
df.head()
```

### 4. Splitting Features and Target Variable
```python
X = df.drop('purchased', axis=1)
y = df['purchased']
```

### 5. Splitting the Data into Training and Testing Sets
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
```

### 6. Applying Ordinal Encoding
```python
oe = OrdinalEncoder(categories=[['Poor', 'Average', 'Good'], ['School', 'UG', 'PG']])
oe.fit(X_train)
X_train = oe.transform(X_train)
X_test = oe.transform(X_test)
```

### 7. Applying Label Encoding
```python
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)
```

## Output Shapes
After encoding, the dataset has the following dimensions:
- `X_train.shape` and `X_test.shape` - Feature matrices after ordinal encoding.
- `y_train.shape` and `y_test.shape` - Target arrays after label encoding.

## Usage
To run the script, execute the following command:
```bash
python encode_data.py
```

## Contributing
Feel free to contribute to this project by adding improvements or new encoding techniques.


---
**Author:** Muhammad Faizan

