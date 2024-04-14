# import scripts
import params

# import libraries
import pandas as pd
import statsmodels.formula.api as smf

# # Load the data with proper date parsing
# merged_data = pd.read_csv("Merged_data.csv", parse_dates=['Date'])

# # Add new calculated columns
# merged_data['delta_D_high'] = merged_data['High.y'] - merged_data['Open.y']
# merged_data['delta_C_high'] = merged_data['High.x'] - merged_data['Open.x']

# Function to select model type and fit the model
def fit_models(data, model_type):

    # Function to fit linear models
    def fit_model(data, response, predictors):
        formula = f"{response} ~ " + " + ".join(predictors)
        model = smf.ols(formula, data=data).fit()
        return model


    predictors_D = ['lag_delta_D', 'lag_delta_D_squared', 'lag2_delta_D_squared']
    predictors_C = ['lag_delta_C', 'lag_delta_C_squared', 'lag2_delta_C_squared']
    
    model_list = {
        'FlexMom_Dl': fit_model(data, 'delta_D', predictors_D),
        'FlexMom_high_Dl': fit_model(data, 'delta_D_high', predictors_D),
        'FlexMom_Cr': fit_model(data, 'delta_C', predictors_C),
        'FlexMom_high_Cr': fit_model(data, 'delta_C_high', predictors_C)
    }
    
    return model_list[model_type]

# # Example usage: Fit and display summary of the 'FlexMom_high_Cr' model using function method
# adjusted_data = merged_data.iloc[:params.trainTestSplit]
# outCoefficients = fit_models(adjusted_data, params.methodType)

# # Print coefficients of the FlexMom_high_Dl model
# print("Coefficients of FlexMom_high_Dl adjusted model:", outCoefficients.params.tolist())
