import pandas as pd
import statsmodels.formula.api as smf

# Load the data with proper date parsing
merged_data = pd.read_csv("Merged_data.csv", parse_dates=['Date'], dayfirst=True)

# Add new calculated columns
merged_data['delta_D_high'] = merged_data['High.y'] - merged_data['Open.y']
merged_data['delta_C_high'] = merged_data['High.x'] - merged_data['Open.x']

# Function to fit linear models
def fit_model(data, response, predictors):
    formula = f"{response} ~ {' + '.join(predictors)}"
    model = smf.ols(formula, data=data).fit()
    return model

# Define predictors
predictors_D = ["lag_delta_D", "lag_delta_D_squared", "lag2_delta_D_squared"]
predictors_C = ["lag_delta_C", "lag_delta_C_squared", "lag2_delta_C_squared"]

# Fit models for the full dataset
FlexMom_Dl = fit_model(merged_data, "delta_D", predictors_D)
FlexMom_high_Dl = fit_model(merged_data, "delta_D_high", predictors_D)

FlexMom_Cr = fit_model(merged_data, "delta_C", predictors_C)
FlexMom_high_Cr = fit_model(merged_data, "delta_C_high", predictors_C)

# Fit models for the adjusted dataset
adjusted_data = merged_data.iloc[:1350]
FlexMom_Cr_adjusted = fit_model(adjusted_data, "delta_C", predictors_C)
FlexMom_high_Cr_adjusted = fit_model(adjusted_data, "delta_C_high", predictors_C)

# Display summary of the high delta C model for adjusted data
print(FlexMom_high_Cr_adjusted.summary())
