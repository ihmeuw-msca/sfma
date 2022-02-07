library(data.table)
library(magrittr)

years <- as.character(1960:2020)

gdp <- read.csv("data/gdp.csv") %>% data.table
gdp <- gdp[, c("Country.Name", paste0("X", years)), with=F]
gdp <- melt(gdp, id.vars="Country.Name")
gdp[, variable := lapply(.SD, function(x) gsub("X", "", x)),
    .SDcols="variable"]
gdp[, variable := as.numeric(variable)]
gdp[value == "", value := NA]
gdp[value == "x", value := NA]
gdp[, value := as.numeric(value)]
setnames(gdp, c("country", "year", "gdp"))

le <- read.csv("data/le.csv") %>% data.table
le <- le[, c("Country.Name", paste0("X", years)), with=F]
le <- melt(le, id.vars="Country.Name")
le[, variable := lapply(.SD, function(x) gsub("X", "", x)),
   .SDcols="variable"]
le[, variable := as.numeric(variable)]
le[value == "", value := NA]
le[value == "x", value := NA]
le[, value := as.numeric(value)]
setnames(le, c("country", "year", "le"))

df <- merge(gdp, le, by=c("country", "year"))
cobbDouglas <- frontier::sfa(log(le) ~ log(gdp),
                             data=df)
cd <- coef(cobbDouglas)
par(mfrow=c(1, 2))
plot(log(df$le) ~ log(df$gdp))
abline(cd[1], cd[2], col='red')
plot((df$le) ~ (df$gdp))
xspace <- seq(0, max(log(df$gdp), na.rm=T), 0.01)
xspace <- cbind(1, xspace)
log.preds <- xspace %*% cd[1:2]
preds <- exp(log.preds)
lines(preds ~ exp(xspace[, 2]), col='red')
