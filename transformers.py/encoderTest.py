from Encoders import OrdinalEncoder, OneHotEncoder, LabelEncoder

# Sample data
data = ['cat', 'dog', 'fish', 'dog', 'cat', 'fish']

# OrdinalEncoder
ord_enc = OrdinalEncoder()
ord_enc.fit(data)
print(ord_enc.transform(data))  # Output might be: [0 1 2 1 0 2]

# OneHotEncoder
onehot_enc = OneHotEncoder()
onehot_enc.fit(data)
print(onehot_enc.transform(data))
# Output might be:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]

# LabelEncoder
label_enc = LabelEncoder()
label_enc.fit(data)
print(label_enc.transform(data))  # Output might be: [0 1 2 1 0 2]
