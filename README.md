# Monte-Carlo-Simulation

Generate distribution of 0.01 quantile (1% percentile) of 10-days overlapping proportional returns obtained from the 3-years timeseries (750 observations) of 1-day returns. Original timeseries is generated using stable distribution with the following parameters: alpha = 1.7; beta = 0.0; gamma = 1.0; delta = 1.0. Show, either numerically or theoretically, that the chosen number of MonteCarlo trials is sufficient.
Requirements: The task must be solved in R. Program code should be clearly written and well commented. Code must be presented along with report comprising all necessary mathematical expressions, description of results and conclusions.
Hint:
1-day returns:
𝑟𝑟𝑖𝑖1=𝑃𝑃𝑖𝑖+1−𝑃𝑃𝑖𝑖𝑃𝑃𝑖𝑖,𝑖𝑖=1…751 
and Pi is price at i-th day.
n-days returns:
𝑟𝑟𝑖𝑖𝑛𝑛=𝑃𝑃𝑖𝑖+𝑛𝑛−𝑃𝑃𝑖𝑖𝑃𝑃𝑖𝑖. 
