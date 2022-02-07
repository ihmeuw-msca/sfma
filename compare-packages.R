library(frontier)
library(sfa)
library(data.table)
library(magrittr)
library(splines2)

# FRONTIER AND SFA PACKAGES ------------------

data(front41Data)
cobbDouglas <- frontier::sfa(log(output) ~ log(capital),
                             data=front41Data)
cobbDouglas2 <- sfa::sfa(log(output) ~ log(capital),
                         data=front41Data)
spline.front <- frontier::sfa(output ~ bSpline(capital),
                              data=front41Data)

plot(log(front41Data$output) ~ log(front41Data$labour))

summary(cobbDouglas)
summary(cobbDouglas2)
capital <- seq(min(front41Data$capital), max(front41Data$capital), 0.25)

pred.data <- data.table(
  intercept=1,
  log_capital=log(capital)
)

vals <- (as.matrix(pred.data) %*% coef(cobbDouglas)[1:2]) %>% exp
vals2 <- (as.matrix(pred.data) %*% coef(cobbDouglas2)[1:2]) %>% exp
vals3 <- predict(spline.front, asInData=T)

newvals3 <- data.table(pred=vals3, capital=front41Data$capital)
setorder(newvals3, capital)

plot(capital, vals, type='l', col='blue')
lines(capital, vals2, type='l', col='red')
points(front41Data$capital, front41Data$output)
lines(newvals3$capital, newvals3$pred, type='l', col='green')

# PHILLIPPINES
data( riceProdPhil )
plot(riceProdPhil)
df <- riceProdPhil %>% data.table
dd <- df[, list(PROD, LABOR, NPK, AREA)]
dd[, PRODLOG := log(PROD)]
dd[, LABORLOG := log(LABOR)]
dd[, NPKLOG := log(NPK)]
dd[, AREALOG := log(AREA)]
dd[, AREA := NULL]
dd[, PROD := NULL]
dd[, LABOR := NULL]
dd[, NPK := NULL]
plot(dd)

cobbDouglas <- frontier::sfa(log(PROD) ~ log(LABOR),
                             data=riceProdPhil)
preds <- predict(cobbDouglas, asInData = T)
plot(exp(preds) ~ riceProdPhil$LABOR)
points(riceProdPhil$PROD ~ riceProdPhil$LABOR, col='red')
