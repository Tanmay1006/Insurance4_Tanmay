import pandas as pd
import numpy as np

def generate_insurance_data(n=1500, seed=42):
    np.random.seed(seed)

    age = np.random.randint(22, 70, n)
    gender = np.random.choice(['Male', 'Female'], n, p=[0.55, 0.45])
    income = np.random.choice(['Low', 'Medium', 'High'], n, p=[0.35, 0.40, 0.25])
    policy_type = np.random.choice(['Health', 'Life', 'Motor', 'Property'], n, p=[0.30, 0.25, 0.25, 0.20])
    region = np.random.choice(['North', 'South', 'East', 'West'], n)
    settlement_team = np.random.choice(['Team A', 'Team B', 'Team C', 'Team D'], n)
    claim_amount = np.round(np.random.exponential(scale=50000, size=n), 2)
    premium_amount = np.round(np.random.uniform(5000, 30000, n), 2)
    claim_age_days = np.random.randint(1, 365, n)
    previous_claims = np.random.randint(0, 6, n)
    documents_submitted = np.random.choice([0, 1], n, p=[0.15, 0.85])
    customer_tenure_years = np.random.randint(1, 20, n)

    # Inject bias: older people + low income + Team D have lower approval
    bias_score = np.zeros(n)
    bias_score += np.where(age > 55, -0.3, 0.1)
    bias_score += np.where(income == 'Low', -0.4, np.where(income == 'High', 0.3, 0.0))
    bias_score += np.where(settlement_team == 'Team D', -0.5, 0.0)
    bias_score += np.where(gender == 'Female', -0.1, 0.05)
    bias_score += np.where(documents_submitted == 1, 0.5, -0.5)
    bias_score += np.where(previous_claims > 3, -0.2, 0.1)
    bias_score += np.where(claim_amount > 100000, -0.2, 0.0)

    prob_approved = 1 / (1 + np.exp(-bias_score))
    policy_status = np.where(np.random.rand(n) < prob_approved, 'Approved', 'Rejected')

    df = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Income_Level': income,
        'Policy_Type': policy_type,
        'Region': region,
        'Settlement_Team': settlement_team,
        'Claim_Amount': claim_amount,
        'Premium_Amount': premium_amount,
        'Claim_Age_Days': claim_age_days,
        'Previous_Claims': previous_claims,
        'Documents_Submitted': documents_submitted,
        'Customer_Tenure_Years': customer_tenure_years,
        'Policy_Status': policy_status
    })
    return df

if __name__ == "__main__":
    df = generate_insurance_data()
    df.to_csv("sample_insurance_data.csv", index=False)
    print(df.head())
    print(df['Policy_Status'].value_counts())
