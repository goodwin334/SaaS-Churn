-- Create SILVER schema
CREATE SCHEMA IF NOT EXISTS SAAS_R.SILVER;

-- 1. Clean Accounts Table
CREATE OR REPLACE TABLE SAAS_R.SILVER.ACCOUNTS AS
SELECT 
    "account_id" as ACCOUNT_ID,
    "account_name" as ACCOUNT_NAME,
    "industry" as INDUSTRY,
    "country" as COUNTRY,
    TO_DATE("signup_date") as SIGNUP_DATE,
    "referral_source" as REFERRAL_SOURCE,
    "plan_tier" as PLAN_TIER,
    "seats" as SEATS,
    "is_trial" as IS_TRIAL,
    "churn_flag" as CHURN_FLAG,
    YEAR(TO_DATE("signup_date")) as SIGNUP_YEAR,
    QUARTER(TO_DATE("signup_date")) as SIGNUP_QUARTER,
    MONTH(TO_DATE("signup_date")) as SIGNUP_MONTH
FROM SAAS_R.BRONZE.ACCOUNTS_RAW;

-- 2. Clean Subscriptions Table
CREATE OR REPLACE TABLE SAAS_R.SILVER.SUBSCRIPTIONS AS
SELECT 
    "subscription_id" as SUBSCRIPTION_ID,
    "account_id" as ACCOUNT_ID,
    TO_DATE("start_date") as START_DATE,
    TO_DATE("end_date") as END_DATE,
    "plan_tier" as PLAN_TIER,
    "seats" as SEATS,
    "mrr_amount" as MRR_AMOUNT,
    "arr_amount" as ARR_AMOUNT,
    "is_trial" as IS_TRIAL,
    "upgrade_flag" as UPGRADE_FLAG,
    "downgrade_flag" as DOWNGRADE_FLAG,
    "churn_flag" as CHURN_FLAG,
    "billing_frequency" as BILLING_FREQUENCY,
    "auto_renew_flag" as AUTO_RENEW_FLAG,
    DATEDIFF(day, TO_DATE("start_date"), COALESCE(TO_DATE("end_date"), CURRENT_DATE())) as SUBSCRIPTION_DAYS,
    CASE WHEN "end_date" IS NULL THEN TRUE ELSE FALSE END as IS_ACTIVE
FROM SAAS_R.BRONZE.SUBSCRIPTIONS_RAW;

-- 3. Clean Support Tickets Table
CREATE OR REPLACE TABLE SAAS_R.SILVER.SUPPORT_TICKETS AS
SELECT 
    "ticket_id" as TICKET_ID,
    "account_id" as ACCOUNT_ID,
    TO_TIMESTAMP("submitted_at") as SUBMITTED_AT,
    TO_TIMESTAMP("closed_at") as CLOSED_AT,
    "resolution_time_hours" as RESOLUTION_TIME_HOURS,
    "priority" as PRIORITY,
    "first_response_time_minutes" as FIRST_RESPONSE_TIME_MINUTES,
    "satisfaction_score" as SATISFACTION_SCORE,
    "escalation_flag" as ESCALATION_FLAG,
    DATE(TO_TIMESTAMP("submitted_at")) as SUBMITTED_DATE,
    HOUR(TO_TIMESTAMP("submitted_at")) as SUBMITTED_HOUR,
    DAYOFWEEK(TO_TIMESTAMP("submitted_at")) as SUBMITTED_DAY_OF_WEEK
FROM SAAS_R.BRONZE.SUPPORT_TICKETS_RAW;

-- 4. Clean Feature Usage Table
CREATE OR REPLACE TABLE SAAS_R.SILVER.FEATURE_USAGE AS
SELECT 
    "usage_id" as USAGE_ID,
    "subscription_id" as SUBSCRIPTION_ID,
    TO_DATE("usage_date") as USAGE_DATE,
    "feature_name" as FEATURE_NAME,
    "usage_count" as USAGE_COUNT,
    "usage_duration_secs" as USAGE_DURATION_SECS,
    "error_count" as ERROR_COUNT,
    "is_beta_feature" as IS_BETA_FEATURE,
    CASE WHEN "error_count" > 0 THEN TRUE ELSE FALSE END as HAS_ERRORS,
    "usage_duration_secs" / 60.0 as USAGE_DURATION_MINUTES
FROM SAAS_R.BRONZE.FEATURE_USAGE_RAW;

-- 5. Clean Churn Events Table
CREATE OR REPLACE TABLE SAAS_R.SILVER.CHURN_EVENTS AS
SELECT 
    "churn_event_id" as CHURN_EVENT_ID,
    "account_id" as ACCOUNT_ID,
    TO_DATE("churn_date") as CHURN_DATE,
    "reason_code" as REASON_CODE,
    "refund_amount_usd" as REFUND_AMOUNT_USD,
    "preceding_upgrade_flag" as PRECEDING_UPGRADE_FLAG,
    "preceding_downgrade_flag" as PRECEDING_DOWNGRADE_FLAG,
    "is_reactivation" as IS_REACTIVATION,
    "feedback_text" as FEEDBACK_TEXT,
    YEAR(TO_DATE("churn_date")) as CHURN_YEAR,
    MONTH(TO_DATE("churn_date")) as CHURN_MONTH
FROM SAAS_R.BRONZE.CHURN_EVENTS_RAW;


SELECT 'ACCOUNTS' as table_name, COUNT(*) as row_count FROM SAAS_R.SILVER.ACCOUNTS
UNION ALL
SELECT 'SUBSCRIPTIONS', COUNT(*) FROM SAAS_R.SILVER.SUBSCRIPTIONS
UNION ALL
SELECT 'SUPPORT_TICKETS', COUNT(*) FROM SAAS_R.SILVER.SUPPORT_TICKETS
UNION ALL
SELECT 'FEATURE_USAGE', COUNT(*) FROM SAAS_R.SILVER.FEATURE_USAGE
UNION ALL
SELECT 'CHURN_EVENTS', COUNT(*) FROM SAAS_R.SILVER.CHURN_EVENTS;

SELECT * FROM SAAS_R.SILVER.ACCOUNTS LIMIT 3;
SELECT * FROM SAAS_R.SILVER.SUBSCRIPTIONS LIMIT 3;
SELECT * FROM SAAS_R.SILVER.SUPPORT_TICKETS LIMIT 3;
SELECT * FROM SAAS_R.SILVER.FEATURE_USAGE LIMIT 3;
SELECT * FROM SAAS_R.SILVER.CHURN_EVENTS LIMIT 3;