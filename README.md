# SaaS Churn Prediction System

Machine learning-based churn prediction system for SaaS accounts using Snowflake and Python.

## Project Structure
```
saas-churn-prediction/
├── python/                     # Python ML scripts
├── churn_prediction_queries/   # Churn analysis and visualization queries
├── data_stages_queries/        # Data staging queries
├── raw_data/                   # Raw data
├── gold_level_data_analysis/   # Final stage data analysis
└── README.md
```

## Database Architecture

### Bronze Layer (Raw Data)
- `ACCOUNTS_RAW`
- `CHURN_EVENTS_RAW`
- `FEATURE_USAGE_RAW`
- `SUBSCRIPTIONS_RAW`
- `SUPPORT_TICKETS_RAW`

### Silver Layer (Cleaned & Transformed)
- `ACCOUNTS`
- `SUBSCRIPTIONS`
- `CHURN_EVENTS`
- `SUPPORT_TICKETS`
- `FEATURE_USAGE`

### Gold Layer (Analytics & ML)
- `CHURN_PREDICTIONS` - ML model predictions
- `CHURN_PREDICTION_FEATURES` - Feature engineering for ML
- `BETA_FEATURE_ADOPTION` - Feature adoption metrics
- `PERFORMANCE_BY_TIER_SEATS` - Performance analysis
- `REVENUE_COHORT_ANALYSIS` - Revenue cohort tracking
- `SUPPORT_WORKLOAD_METRICS` - Support team metrics
- `UPGRADE_FUNNEL_BY_INDUSTRY` - Upgrade conversion analysis

## Setup Instructions

### 1. Database Setup
Run the SQL scripts in this order:
```sql
-- 1. Create database and schemas
sql/data_stages_queries/db_create.sql

-- 2. Create raw tables
"sql/data_stages_queries/raw_table_create.sql"

-- 3. Create silver layer
"sql/data_stages_queries/silver_layer_create.sql"

-- 4. Create gold layer
"sql/data_stages_queries/gold_layer_create.sql"
```

### 2. Load Data
Load your raw data into the Bronze layer tables.

### 3. Run Churn Prediction Model
```bash
python python/churn_prediction.py
```

This will:
- Pull data from Gold layer
- Train ML model
- Generate predictions
- Write results to `GOLD.CHURN_PREDICTIONS` and `GOLD.CHURN_PREDICTION_FEATURES`

## Analysis Queries

Located in `sql/analysis/`:

### Main Reports
- `churn_risk_by_country.sql` - Predicted churn risk by country
- `churn_risk_by_industry.sql` - Industry analysis
- `churn_risk_vs_errors.sql` - Churn risk vs. errors
- `churn_risk_vs_feature_adoption.sql` - Churn risk vs. feature adoption
- `churn_top_20_high_risk_accounts.sql` - Churn risk for the top 20 accounts

### Dashboards
Run these queries in Snowflake and use the Chart tab for visualization:
- Revenue at Risk
- Customer Health Scores
- Feature Adoption vs Churn
- Support Ticket Impact

## Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: Support tickets, satisfaction scores, feature usage, errors, account age
- **Output**: Churn probability (0-100%) and binary prediction (TRUE/FALSE)
- **Refresh**: Run `churn_prediction.py` to update predictions

## Key Metrics

- **Churn Risk %**: Probability of account churning
- **Revenue at Risk**: MRR/ARR from high-risk accounts
- **Action Priority**: Categorization based on risk + value

## Requirements

- Snowflake account with appropriate permissions
- Python 3.8+
- Libraries: pandas, scikit-learn, snowflake-connector-python

## Contributing

1. Create feature branch
2. Make changes
3. Test thoroughly
4. Submit pull request

## License

Internal use only