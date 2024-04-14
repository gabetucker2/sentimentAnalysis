#############################################################
# * STATIC
datasetName = "data.csv"

daysInTradeYear = 252
lossSellPercent = 0.02
startMoney = 10000

leverage = True
leverageMultiplier = 2

epochs = 1

dampenerMult = 0.8

numTrainYears = 3
numTestYears = 1

methodType = "FlexMom_high_Dl"
# FlexMom_Dl
# FlexMom_high_Dl
# FlexMom_Cr
# FlexMom_high_Cr

#############################################################
# * PROCEDURAL

trainTestSplit = daysInTradeYear * numTrainYears
