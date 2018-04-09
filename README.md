# Modeling trend in variance of spatio-temporal data

There is a considerable interest in determining if there is an increasing trend in the climate variability. An increase in the temperature variability will increase the
probability of extreme hot outliers. It might be harder for the society to adapt to these extremes than to the gradual increase in the mean temperature.
In this project, we consider the problem of detecting the trend in the temperature
volatility. All analyses are performed on a sub-set of the European Centre for Medium-Range Weather Forecasts (ECMWF) ERA-40 dataset. This dataset include the
temperature measurements over a grid over the earth from 1957 to 2002.

The main contribution of this work is to develop a new methodology for detecting
the trend in the volatility of spatio-temporal data. In this methodology, the variance at each position and time, is considered as a hidden (unobserved) variable. The value of these hidden variables are then estimated by maximizing the likelihood of the observed data. We show that this formulation per se, is not appropriate for detecting the trend.

To overcome this issue, we penalize the differences between the estimated variances of the observations which are temporally and/or spatially close to each other. This will result in an optimization problem called the generalized LASSO problem. The dimension of this optimization problem is very high. In addition, the objective function is not of the form of a sum of square problem. Therefore the standard methods for solving the generalized LASSO cannot be applied directly. We investigate two methods for solving this optimization problem. In the first method, we adopt an optimization technique called alternative direction method of multipliers (ADMM), to divide the total problem into several sub-problems of much lower dimension and show how the total problem can be solved by iteratively solving these sub-problems. The second method, called the linearized ADMM algorithm solves the main problem by iteratively solving a linearized version of it. We will compare the benefits of each method.

Read the paper [here](https://github.com/akhodadadi/VolatilityTrend/blob/master/working_paper.pdf).
