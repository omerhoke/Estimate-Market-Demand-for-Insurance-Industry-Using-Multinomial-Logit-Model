# =====================================================
# INSURANCE DEMAND — MULTINOMIAL LOGIT
# 4 SPECS + ELASTICITIES + EXCEL OUTPUT
# =====================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm

# -----------------------------------------------------
# 1. LOAD DATA (LONG FORMAT)
# -----------------------------------------------------
df = pd.read_csv("insurance_choice_data.csv")

# Required columns (example):
# consumer_id, policy_id, choice (0/1), price, coverage_level,
# deductible, coverage_features, age, income, education,
# risk_score, state, year

# -----------------------------------------------------
# 2. CATEGORICAL VARIABLES
# -----------------------------------------------------
df['policy_id'] = df['policy_id'].astype('category')
df['state'] = df['state'].astype('category')
df['year'] = df['year'].astype('category')

# -----------------------------------------------------
# 3. INTERACTION VARIABLES (BEHAVIORAL)
# -----------------------------------------------------
df['price_income'] = df['price'] * df['income']
df['deductible_risk'] = df['deductible'] * df['risk_score']

# -----------------------------------------------------
# 4. BASE REGRESSORS
# -----------------------------------------------------
base_vars = [
    'price',
    'coverage_level',
    'deductible',
    'coverage_features',
    'age',
    'income',
    'education',
    'risk_score',
    'price_income',
    'deductible_risk'
]

# -----------------------------------------------------
# 5. MULTINOMIAL LOGIT FUNCTION
# -----------------------------------------------------
def run_mnl(data, regressors, model_name):
    X = data[regressors]
    X = sm.add_constant(X, has_constant='add')
    y = data['choice']

    model = sm.MNLogit(y, X)
    result = model.fit(method='newton', maxiter=100, disp=False)

    print(f"\n========== {model_name} ==========")
    print(result.summary())

    return result

# -----------------------------------------------------
# 6. MODEL 1 — BASELINE
# -----------------------------------------------------
mnl_1 = run_mnl(df, base_vars, "Model 1: Baseline")

# -----------------------------------------------------
# 7. MODEL 2 — ADD TIME FIXED EFFECTS
# -----------------------------------------------------
year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True)
df_2 = pd.concat([df, year_dummies], axis=1)

vars_2 = base_vars + list(year_dummies.columns)
mnl_2 = run_mnl(df_2, vars_2, "Model 2: + Time FE")

# -----------------------------------------------------
# 8. MODEL 3 — ADD STATE FIXED EFFECTS
# -----------------------------------------------------
state_dummies = pd.get_dummies(df['state'], prefix='state', drop_first=True)
df_3 = pd.concat([df_2, state_dummies], axis=1)

vars_3 = vars_2 + list(state_dummies.columns)
mnl_3 = run_mnl(df_3, vars_3, "Model 3: + State FE")

# -----------------------------------------------------
# 9. MODEL 4 — ADD STATE-SPECIFIC TIME TRENDS
# -----------------------------------------------------
df_3['year_num'] = df_3['year'].cat.codes

trend_vars = []
for s in state_dummies.columns:
    trend_name = f"{s}_trend"
    df_3[trend_name] = df_3[s] * df_3['year_num']
    trend_vars.append(trend_name)

vars_4 = vars_3 + trend_vars
mnl_4 = run_mnl(df_3, vars_4, "Model 4: + FE + Trends")

# -----------------------------------------------------
# 10. PRICE ELASTICITY CALCULATION
# -----------------------------------------------------
def compute_price_elasticities(model, data, price_var='price'):
    X = model.model.exog
    probs = model.predict(X)

    beta_price = model.params.loc[price_var]
    price = data[price_var].values

    elasticities = beta_price.values * price[:, None] * (1 - probs)
    return pd.DataFrame(elasticities, columns=model.params.columns)

elasticities_m4 = compute_price_elasticities(mnl_4, df_3)

print("\nMean Own-Price Elasticities (Model 4):")
print(elasticities_m4.mean())

# -----------------------------------------------------
# 11. EXPORT RESULTS TO EXCEL
# -----------------------------------------------------
with pd.ExcelWriter("insurance_mnl_results.xlsx", engine="xlsxwriter") as writer:

    def save_coefficients(result, sheet_name):
        coef = result.params
        se = result.bse
        tstat = coef / se
        pval = result.pvalues

        out = pd.concat(
            [coef, se, tstat, pval],
            axis=1,
            keys=["Coefficient", "StdError", "Tstat", "Pvalue"]
        )
        out.to_excel(writer, sheet_name=sheet_name)

    save_coefficients(mnl_1, "Model1_Baseline")
    save_coefficients(mnl_2, "Model2_TimeFE")
    save_coefficients(mnl_3, "Model3_StateFE")
    save_coefficients(mnl_4, "Model4_Trends")

    elasticities_m4.to_excel(
        writer,
        sheet_name="Model4_Price_Elasticities",
        index=False
    )

print("\n✅ All models estimated and results exported to insurance_mnl_results.xlsx")
