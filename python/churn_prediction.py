import snowflake.connector
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Connect to Snowflake
conn = snowflake.connector.connect(
    user= "GOODWIN334",  # Your username
    password= "nxuxSAAjBfS5Sq2",  # Your password
    account= "NHGTEJH-WX15778",  # e.g., 'abc12345.us-east-1'
    warehouse= "COMPUTE_WH",
    database= "SAAS_R",
    schema= "GOLD"
)

print("Connected to Snowflake!")

# Step 2: Load data from Snowflake
query = """
SELECT 
    ACCOUNT_ID,
    INDUSTRY,
    COUNTRY,
    REFERRAL_SOURCE,
    PLAN_TIER,
    ACCOUNT_AGE_DAYS,
    SEATS,
    MRR_AMOUNT,
    ARR_AMOUNT,
    SUBSCRIPTION_DAYS,
    IS_ACTIVE,
    TOTAL_TICKETS,
    AVG_RESOLUTION_TIME,
    AVG_FIRST_RESPONSE_TIME,
    AVG_SATISFACTION_SCORE,
    ESCALATED_TICKETS,
    URGENT_TICKETS,
    UNIQUE_FEATURES_USED,
    TOTAL_USAGE_COUNT,
    TOTAL_USAGE_MINUTES,
    TOTAL_ERRORS,
    HAS_CHURNED
FROM SAAS_R.GOLD.CHURN_PREDICTION_FEATURES
"""

df = pd.read_sql(query, conn)
print(f"Loaded {len(df)} records from Snowflake")
print(f"Churn rate: {df['HAS_CHURNED'].mean():.2%}")

# Step 3: Feature Engineering
print("\nPreparing features...")

# Handle missing values
df = df.fillna({
    'AVG_RESOLUTION_TIME': df['AVG_RESOLUTION_TIME'].median(),
    'AVG_FIRST_RESPONSE_TIME': df['AVG_FIRST_RESPONSE_TIME'].median(),
    'AVG_SATISFACTION_SCORE': df['AVG_SATISFACTION_SCORE'].median(),
    'TOTAL_TICKETS': 0,
    'ESCALATED_TICKETS': 0,
    'URGENT_TICKETS': 0,
    'UNIQUE_FEATURES_USED': 0,
    'TOTAL_USAGE_COUNT': 0,
    'TOTAL_USAGE_MINUTES': 0,
    'TOTAL_ERRORS': 0
})

# Encode categorical variables
le_industry = LabelEncoder()
le_country = LabelEncoder()
le_referral = LabelEncoder()
le_plan = LabelEncoder()

df['INDUSTRY_ENCODED'] = le_industry.fit_transform(df['INDUSTRY'])
df['COUNTRY_ENCODED'] = le_country.fit_transform(df['COUNTRY'])
df['REFERRAL_ENCODED'] = le_referral.fit_transform(df['REFERRAL_SOURCE'])
df['PLAN_TIER_ENCODED'] = le_plan.fit_transform(df['PLAN_TIER'])

# Create additional features
df['ERROR_RATE'] = df['TOTAL_ERRORS'] / (df['TOTAL_USAGE_COUNT'] + 1)
df['TICKETS_PER_DAY'] = df['TOTAL_TICKETS'] / (df['ACCOUNT_AGE_DAYS'] + 1)
df['USAGE_PER_DAY'] = df['TOTAL_USAGE_COUNT'] / (df['ACCOUNT_AGE_DAYS'] + 1)

# Select features for model
feature_cols = [
    'ACCOUNT_AGE_DAYS',
    'SEATS',
    'MRR_AMOUNT',
    'ARR_AMOUNT',
    'SUBSCRIPTION_DAYS',
    'TOTAL_TICKETS',
    'AVG_RESOLUTION_TIME',
    'AVG_FIRST_RESPONSE_TIME',
    'AVG_SATISFACTION_SCORE',
    'ESCALATED_TICKETS',
    'URGENT_TICKETS',
    'UNIQUE_FEATURES_USED',
    'TOTAL_USAGE_COUNT',
    'TOTAL_USAGE_MINUTES',
    'TOTAL_ERRORS',
    'INDUSTRY_ENCODED',
    'COUNTRY_ENCODED',
    'REFERRAL_ENCODED',
    'PLAN_TIER_ENCODED',
    'ERROR_RATE',
    'TICKETS_PER_DAY',
    'USAGE_PER_DAY'
]

X = df[feature_cols]
y = df['HAS_CHURNED']

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Step 5: Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Models
print("\n Training models...")

# Model 1: Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# Model 2: XGBoost
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    scale_pos_weight=len(y_train[y_train==False]) / len(y_train[y_train==True]),
    n_jobs=-1
)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Step 7: Evaluate Models
print("\n" + "="*60)
print("RANDOM FOREST RESULTS")
print("="*60)
print(classification_report(y_test, rf_pred, target_names=['Not Churned', 'Churned']))
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, rf_proba):.3f}")

print("\n" + "="*60)
print("XGBOOST RESULTS")
print("="*60)
print(classification_report(y_test, xgb_pred, target_names=['Not Churned', 'Churned']))
print(f"Accuracy: {accuracy_score(y_test, xgb_pred):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, xgb_proba):.3f}")

# Step 8: Feature Importance
print("\n Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10))

# Visualizations
plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Features for Churn Prediction (XGBoost)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved feature_importance.png")

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned'])
plt.title('Confusion Matrix - XGBoost')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved confusion_matrix.png")

# Step 9: Score All Active Accounts
print("\n Scoring all active accounts...")
X_all_scaled = scaler.transform(df[feature_cols])
df['CHURN_PROBABILITY'] = xgb_model.predict_proba(X_all_scaled)[:, 1]
df['CHURN_PREDICTION'] = xgb_model.predict(X_all_scaled)

# Identify high-risk accounts
high_risk = df[
    (df['HAS_CHURNED'] == False) & 
    (df['CHURN_PROBABILITY'] > 0.6)
].sort_values('MRR_AMOUNT', ascending=False)

print(f"\n {len(high_risk)} high-risk accounts identified!")
print("\nTop 10 High-Risk Accounts by MRR:")
print(high_risk[['ACCOUNT_ID', 'INDUSTRY', 'PLAN_TIER', 'MRR_AMOUNT', 'CHURN_PROBABILITY']].head(10).to_string(index=False))

# Step 10: Push Predictions Back to Snowflake
print("\n Uploading predictions to Snowflake...")

# Create predictions table
predictions_df = df[['ACCOUNT_ID', 'CHURN_PROBABILITY', 'CHURN_PREDICTION']].copy()

# Write to Snowflake
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE OR REPLACE TABLE SAAS_R.GOLD.CHURN_PREDICTIONS (
    ACCOUNT_ID VARCHAR,
    CHURN_PROBABILITY FLOAT,
    CHURN_PREDICTION BOOLEAN,
    PREDICTION_DATE DATE
)
""")

# Insert predictions
for _, row in predictions_df.iterrows():
    cursor.execute(f"""
    INSERT INTO SAAS_R.GOLD.CHURN_PREDICTIONS 
    VALUES ('{row['ACCOUNT_ID']}', {row['CHURN_PROBABILITY']}, {row['CHURN_PREDICTION']}, CURRENT_DATE())
    """)

conn.commit()
print(" Predictions uploaded to SAAS_R.GOLD.CHURN_PREDICTIONS")

# Close connection
cursor.close()
conn.close()

print("\n Churn prediction model complete!")