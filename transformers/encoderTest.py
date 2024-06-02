from Encoders import OrdinalEncoder, OneHotEncoder, LabelEncoder

# Sample data
data = ['cat', 'dog', 'fish', 'dog', 'cat', 'fish']

# OrdinalEncoder
ord_enc = OrdinalEncoder()
ord_enc.fit(data)
print(ord_enc.transform(data))  # Output might be: [0 1 2 1 0 2]
"""
OrdinalEncoder:
- Fits the encoder on the data and transforms it into ordinal values.
- Unique values in the data are assigned integer indices.
- Example output: [0 1 2 1 0 2]
"""

# OneHotEncoder
onehot_enc = OneHotEncoder()
onehot_enc.fit(data)
print(onehot_enc.transform(data))
"""
OneHotEncoder:
- Fits the encoder on the data and transforms it into one-hot encoded values.
- Each unique value in the data is represented as a binary vector.
- Example output:
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]]
"""

# LabelEncoder
label_enc = LabelEncoder()
label_enc.fit(data)
print(label_enc.transform(data))  # Output might be: [0 1 2 1 0 2]
"""
LabelEncoder:
- Fits the encoder on the data and transforms it into label encoded values.
- Each unique value in the data is assigned a unique integer label.
- Example output: [0 1 2 1 0 2]
"""
