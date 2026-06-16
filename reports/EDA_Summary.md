# EDA Summary – Fraud Detection

## Dataset Overview
- E-commerce: 151,112 transactions, 9.36% fraud rate
- Credit Card: 284,807 transactions, 0.17% fraud rate

## Key Distributions
1. Purchase Value: Right-skewed; fraud clusters at lower values (£50–£200)
2. Age: Fraud slightly higher in 25–35 age group
3. Hour of Day: Fraud peaks between 1 AM – 5 AM (nocturnal attacks)
4. Day of Week: Higher fraud on Sundays and Mondays
5. Device Count: Fraud rings use shared devices (3+ users per device)

## Feature Engineering Rationale
| Feature | Business Meaning | Why It Helps |
|---------|------------------|--------------|
| device_count | Number of users per device | Fraud rings reuse devices |
| time_since_signup_hours | Time between signup and purchase | Bots transact immediately |
| purchase_hour | Hour of day | Late night = higher risk |
| purchase_dayofweek | Day of week | Weekend patterns differ |

## Class Imbalance
- SMOTE applied only on training set to avoid data leakage
- Before SMOTE: 90.6% legitimate / 9.4% fraud
- After SMOTE: 50% / 50% (balanced)
