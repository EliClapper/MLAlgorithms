metrics <- function(obs, pred){
  RMSE <- sqrt(mean((obs-pred)^2))
  MAD <- mean(abs(obs-pred))
  MaxDev <- max(abs(obs-pred))
  MinDev <- min(abs(obs-pred))
  R2 <- 1-(sum((obs-pred)^2)/sum((obs - mean(obs))^2))
  return(c(RMSE = RMSE, MAD = MAD, MaxDev = MaxDev, MinDev = MinDev, R2 = R2))
}