#performance metrics
metrics <- function(obs, pred){
  RMSE <- sqrt(mean((obs-pred)^2))
  MAD <- mean(abs(obs-pred))
  MaxDev <- max(abs(obs-pred))
  MinDev <- min(abs(obs-pred))
  R2 <- 1-((sum((obs-pred)^2))/(sum((obs - mean(obs))^2)))
  return(c(RMSE = RMSE, MAD = MAD, MaxDev = MaxDev, MinDev = MinDev, R2 = R2))
}

#OLS multilinear
OLS <- function(y, x){
  n <- length(y)         #number of observations
  nx <- ifelse(is.null(ncol(x)), length(x), nrow(x))
  
  if(nx != n){stop("lengths y and x differ")}
  
  x <- cbind(rep(1,n),x) #add intercept
  dfs <- n - ncol(x)     #degrees of freedom
  
  weights <- solve(t(x)%*%x) %*% t(x) %*% y #obtain weights using linear algebra
  
  y_pred <- weights[1] + rowSums(t(t(x[,2:ncol(x)])*weights[2:ncol(x)])) # obtain predicted values
  e <- y - y_pred # the errors to obtain Standard Errors of the estimates
  var_res <- as.numeric(t(e)%*%e/dfs) #estimate for residual variance
  varcov_weights <- var_res  * solve(t(x)%*%x)
  ses <- sqrt(diag(varcov_weights))
  
  t_vals <- weights/ses # obtain t-values
  p_vals <- pt(abs(t_vals), dfs, lower.tail = F) * 2 #probability of this t-value against central t-distribution with Df = dfs, two-tailed
  
  metricz <- metrics(y, y_pred) #obtain metrics of performance
  estimates <- cbind(weights, ses, t_vals, p_vals)
  colnames(estimates) <- c("weights", 'SEs', 't-value', 'p-value')
  rownames(estimates) <- c("b0", sprintf("b%d", 1:(ncol(x)-1)))
  
  return(list(estimates = estimates, `degrees of freedom` = dfs, metrics = metricz)) #return list
}

#Gradient Descent Multilinear
GD_Multi <- function(Y, x, l_rate = 1E-3, thresh = 1, maxit = 5000){
  
  n <- length(Y)
  Nx <- ifelse(is.null(ncol(x)), length(x), nrow(x))
  nx <- ifelse(is.null(ncol(x)), 1, ncol(x))    #number of predictors
  
  weights <- matrix(NA, maxit, nx+1)                    #create memory for the weights
  colnames(weights) <- c("b0", sprintf('b%d', 1:nx))    #give names
  weights[1,] <- replicate(nx+1, runif(1))              #set initial values
  
  y_pred <- weights[1,1] + rowSums(t(t(x)*weights[1,2:(nx+1)])) #calculate fitted values using initial weights
  res <- Y-y_pred
  MSE <- mean((res)^2)                                  #set initial MSE with random values
  iter <- 1                                             #set counter
  
  Dbs <- matrix(NA, maxit, nx+1)                        #set matrix for derivatives
  
  while(MSE > thresh){
    Dbs[iter, 1] <- -2*mean(res) #derivative w.r.t b0
    for(i in 2:(nx+1)){                   #derivative w.r.t weights
      Dbs[iter, i] <- -2* mean(x[, (i-1)]*res)
    }
    
    #this is the same as above.
    # Dbs[iter,1] <- (2/n) * sum(-(res))
    # for(i in 2:(nx+1)){                   
    #   Dbs[iter, i] <- (2/n) * sum(-x[,i-1]*res)
    # }
    
    weights[(iter+1),1] <- weights[iter,1] - l_rate*Dbs[iter,1] #update intercept
    weights[(iter+1),2:(nx+1)] <- weights[iter,2:(nx+1)] - l_rate*Dbs[iter, 2:(nx+1)] #update regression coefficient
    
    y_pred <- weights[iter,1] + rowSums(t(t(x)*weights[iter,2:(nx+1)])) #update predicted values
    
    res <- Y-y_pred
    MSE <- mean((res)^2) #update MSE
    
    
    iter <- iter + 1 #count iterations up
    
    if(MSE == Inf | MSE == -Inf){
      stop('No convergence, try a smaller learning rate such as dividing the current learning rate by 10')
    }
    
    if(iter == maxit){
      print('max iteration reached')
      return(weights)
    }
  }
  print('solution found')
  return(weights)
  
}