# script imports
import params
import functions

# library imports
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

# unpack data
df = pd.read_csv('data.csv', delimiter='\t')
# Ensure your data has the necessary columns
# Example of preparing additional necessary columns (you will need to adjust this to fit your data specifics)
df['lag_delta_D'] = df['delta_D'].shift(1)
df['lag_delta_D_squared'] = df['lag_delta_D'] ** 2
df['lag2_delta_D_squared'] = df['lag_delta_D'].shift(1) ** 2

# Assuming df is already loaded and ready
traindf = df.iloc[params.trainTestSplit:]
testdf = df.iloc[:params.trainTestSplit]

# Call fit_models with the appropriate model type and data slice
model_outputs = functions.fit_models(traindf, params.methodType)
betas = np.array([model.params for model in model_outputs.values()]) * params.dampenerMult

# ALPHA = 0.51582  * params.dampenerMult
# BETA1 = -0.04540 * params.dampenerMult
# BETA2 = 0.05334  * params.dampenerMult
# BETA3 = 0.03338  * params.dampenerMult

buyPrices = []
sellPrices = []
deltaMoney = 0
workingFinalPrice = 0
moneyOverTime = []

for e in range(params.epochs):

    workingMoney = params.startMoney
    moneyOverTime = [workingMoney] # track money over time

    # sampled_indices = random.sample(range(len(traindf)), params.daysInTradeYear)
    # for idx in sampled_indices: # probablistic

    startDay = 0
    for idx in range(startDay, params.daysInTradeYear*params.numTestYears): # first 252
        
        row = traindf.iloc[idx]  # Get the row corresponding to the index

        open_cl = row['OpenCL']
        high_cl = row['HighCL']
        low_cl = row['LowCL']
        close_cl = row['CloseCL']
        open_da = row['OpenDA']
        high_da = row['HighDA']
        low_da = row['LowDA']
        close_da = row['CloseDA']
        ldd = row['LDD']
        ldds = row['LDDS']
        l2dds = row['L2DDS']

        m2 = np.array([1, ldd, ldds, l2dds])
        
        buyPrice = open_da
        stopPrice = buyPrice * (1-params.lossSellPercent)
        shares = workingMoney / buyPrice

        deltaToHighest = np.dot(betas, m2)

        # deltaToHighest = ALPHA + BETA1*ldd + BETA2*ldds + BETA3*l2dds
        highestEstimate = open_da + deltaToHighest

        sellPrice = 0
        if high_da >= highestEstimate:
            sellPrice = highestEstimate
        elif low_da < stopPrice:
            sellPrice = stopPrice
        else:
            sellPrice = close_da

        deltaShare = sellPrice - buyPrice

        if params.leverage:
            workingMoney += deltaShare*shares*params.leverageMultiplier
        else:
            workingMoney += deltaShare*shares
        moneyOverTime.append(workingMoney)

    workingFinalPrice += moneyOverTime[-1]

averageFinalMoney = workingFinalPrice / params.epochs

print(f"Days: {params.daysInTradeYear}")
print(f"Leverage: {params.leverage}")
print(f"Simulations: {params.epochs}")
print(f"Average final money: {averageFinalMoney}")
print(f"Money gained: {averageFinalMoney - params.startMoney}")
print(f"% money increase: {((averageFinalMoney/params.startMoney)-1)*100}%")

plt.plot(moneyOverTime)
plt.xlabel('Time step')
plt.ylabel('Money')
plt.title('Random Days from the 5 Years After Training')
plt.show()
