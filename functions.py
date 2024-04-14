# import scripts
import params

# import libraries
import pandas as pd
import statsmodels.formula.api as smf

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
