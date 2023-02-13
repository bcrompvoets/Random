library(randomForestSRC)
df1 <- data.frame(read.csv("/Users/breannacrompvoets/Documents/Star_Formation/YSO+Classification/Archive/Phase_4__2MASS_UpperLim_Classification/Scripts/CC_Webb_Predictions_Prob_Dec192022_Spitz_ONLY_YS.csv"))
input <- df1[ , c(5:23, 41)]
input[,c(19,20)] <- lapply(input[ ,c(19,20)] , factor)
o.rfq <- imbalanced(SPICY_Class_0.1 ~ ., data=input,na.action='na.impute')
print(get.auc(input$SPICY_Class_0.1, o.rfq$predicted.oob))