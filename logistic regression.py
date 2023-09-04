import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

# Load the training and test datasets
train_data = pd.read_csv('/content/train.csv', dtype={'yr': str})
test_data = pd.read_csv('/content/test.csv', dtype={'yr': str})


# Define features and target variable
numeric_features = ['GP', 'Min_per', 'Ortg', 'usg', 'eFG', 'TS_per', 'ORB_per', 'DRB_per',
                   'AST_per', 'TO_per', 'FTM', 'FTA', 'FT_per', 'twoPM', 'twoPA', 'twoP_per',
                   'TPM', 'TPA', 'TP_per', 'blk_per', 'stl_per', 'ftr', 'porpag', 'adjoe',
                   'Rec_Rank', 'ast_tov', 'rim_ratio', 'mid_ratio', 'dunks_ratio',
                   'pick', 'drtg', 'adrtg', 'dporpag', 'stops', 'bpm', 'obpm', 'dbpm', 'gbpm',
                   'mp', 'ogbpm', 'dgbpm', 'oreb', 'dreb', 'treb', 'ast', 'stl', 'blk', 'pts']

target = 'drafted'

# Split data into features and target
X = train_data[numeric_features]
y = train_data[target]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a SimpleImputer to fill missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

# Preprocess numeric features using standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)

# Initialize and train a logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)  # or a higher value
model.fit(X_train_scaled, y_train)

# Predict probabilities on the validation set
y_val_pred_prob = model.predict_proba(X_val_scaled)[:, 1]

# Calculate AUROC score
auroc_score = roc_auc_score(y_val, y_val_pred_prob)
print(f'AUROC Score: {auroc_score:.4f}')

# Preprocess
X_test = test_data[numeric_features]
X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)

# Predict probabilities on the test set
y_test_pred_prob = model.predict_proba(X_test_scaled)[:, 1]


submission = pd.DataFrame({'player_id': test_data['player_id'], 'drafted': y_test_pred_prob})
submission.to_csv('submission.csv', index=False)
