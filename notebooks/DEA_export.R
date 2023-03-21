library(dplyr)
library(Benchmarking)
library(tidyverse)
#############################################
df <- read.csv('../data/data_var.csv')
df_models <- read.csv('../data/data_mod.csv')
dim(df)
##############################################
df_homosk <- df %>%
  filter(data_type == 'homosk')

df_heterosk <- df %>%
  filter(data_type == 'heterosk')

dea_func <- function(df){
  x = df$x
  y = df$y
  y_true = df$y_true
  front <- dea(df$x,df$y,SLACK=TRUE)
  front <- front$eff
  # colnames(df_new) = c('data_id','front_dea')
  df_new <- df[c("data_id","x","y","data_type")]
  df_new$model <- "dea"
  value <- front
  df_model_dea <- cbind(df_new,value)
  return (df_model_dea)
  }
df_homosk_ <- dea_func(df_homosk)
df_heterosk_ <- dea_func(df_heterosk)
df_model_dea <- rbind(df_homosk_,df_heterosk_)
df_mod_with_dea <- rbind(df_models,df_model_dea)

##########################################################
# Export to csv:
#############################################################
# write.csv(df_mod_with_dea, "../data/df_mod_inc_dea.csv", row.names=FALSE)

write.csv(df_mod_with_dea, "../data/df_mod_inc_dea.csv", row.names=FALSE)






