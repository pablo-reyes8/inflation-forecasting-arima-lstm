
*****************************************************
*  Inflation Forecasting Diagnostics & Modeling
*  A Comprehensive Stata Script
*
*  "Analyzing and Forecasting Inflation in Maryland
*   using ARIMA Methodology and Diagnostic Tests"
*
*  Key Procedures:
*    - Augmented Dickey–Fuller Stationarity Tests
*    - Autocorrelation Function (ACF) & Partial Autocorrelation Function (PACF) Analysis
*    - Data Differencing to Achieve Stationarity
*    - ARIMA Model Identification and Selection via AIC/BIC
*    - Model Estimation and Diagnostic Checking
*    - Forecast Generation and Accuracy Assessment
*
*  Author: Pablo Reyes
*  Date: June 2025
*****************************************************


// Before we start // 

clear all
set more off

// Import the data // 

import excel "C:\Users\alejo\OneDrive\Escritorio\Universidaad\6 Semestre\Econometria\GitHub Repository\Data Cleaned.xlsx", sheet("Hoja1") firstrow



// Generate the time variable // 

gen time_variable = tq(1978q1) + _n - 1
tsset time_variable, quarterly // Set the variable as time for stata

 
 
*****************************************************
*  Augmented Dickey–Fuller (ADF) Unit Root Test
*
*  We perform the ADF test on the Maryland inflation series (pi)
*  to determine whether the data are stationary or contain a unit root,
*  guiding our choice of differencing for ARIMA modeling.
*
*  1. dfuller pi
*     - Specification: no constant, no trend  
*     - H0: π_t follows a unit-root process (non-stationary)  
*     - H1: π_t is stationary around zero  
*     - Inspect the test statistic and MacKinnon p-value  
*
*  2. dfuller pi, drift
*     - Specification: includes an intercept ("drift")  
*     - H0: π_t has a unit root  
*     - H1: π_t is stationary around a non-zero mean  
*     - Useful if the series exhibits a non-zero long-run average  
*
*  In both cases, a p-value below our significance threshold (e.g. 0.05)
*  leads us to reject the null of non-stationarity.  If neither test rejects,
*  we will difference the series and repeat until stationarity is achieved.
*****************************************************

dfuller pi
dfuller pi , drift

*****************************************************
*  Conclusion:
*    - ADF without drift (p = 0.0113):
*        • Reject H0 at 5% significance (0.0113 < 0.05)
*        • Fail to reject H0 at 1% significance (0.0113 > 0.01)
*        → Series is stationary at the 5% level but not at 1%.
*
*    - ADF with drift (p = 0.0004):
*        • Reject H0 at both 5% and 1% significance (0.0004 < 0.01)
*        → Series is stationary around a non-zero mean under a strict threshold.
*
*  Modeling Strategy:
*    1. Estimate ARIMA models using a 5% significance criterion  
*       for parameter inclusion and diagnostic tests.
*    2. Estimate ARIMA models using a 1% significance criterion  
*       to assess the effect of a more stringent threshold.
*
*  Comparing models across these two significance levels
*  allows us to gauge how threshold choice influences parsimony
*  and forecast performance.
*****************************************************



*****************************************************
*  ARIMA Model Selection at 5% Significance Level
*
*  Identifying the Optimal ARIMA(p,0,q) Specification
*  Using a 5% Parameter Inclusion Criterion
*  Use AC and PAC graphs also a intesive search for parameters p , q 
*****************************************************



*****************************************************
*  Step 1: Model Identification via ACF & PACF
*
*  Following the Box–Jenkins methodology, we first examine
*  the ACF and PACF plots to select appropriate AR and MA
*  orders before estimating any ARIMA models.
*****************************************************

ac pi, name (IPCAC)
pac pi, name (IPCPAC)
graph combine IPCAC IPCPAC, r(2) name(IPCACPAC)


*****************************************************
*  Conclution

*  ACF/PACF Overview & Candidate ARMA Specifications
*  The ACF displays a strong positive spike at lag 1 followed by a slow
*  exponential decay, while the PACF exhibits significant cuts at lags 1
*  and 2 before leveling off.  This mixed pattern of autoregressive and
*  moving-average behavior suggests evaluating the following ARIMA(p,0,q)
*  models:
*    • ARIMA(1,0,1)
*    • ARIMA(1,0,2)
*    • ARIMA(1,0,3)
*    • ARIMA(2,0,1)
*    • ARIMA(2,0,2)
*    • ARIMA(2,0,3)
*****************************************************


*****************************************************
*  Step 2: Model Estimation & AIC Comparison
*
*  We now estimate each candidate ARIMA(p,0,q) model on the
*  Maryland inflation series and extract the Akaike Information
*  Criterion (AIC) to identify the specification that best
*  balances goodness-of-fit and parsimony.
*****************************************************

arima pi, arima(1,0,1)
estat ic

arima pi, arima(1,0,2)
estat ic

arima pi, arima(1,0,3)
estat ic

arima pi, arima(2,0,1)
estat ic

arima pi, arima(2,0,2)
estat ic

arima pi, arima(2,0,3)
estat ic

*****************************************************
*  Conclution
*
*  Among all candidate models, ARIMA(1,0,3) achieves the lowest
*  Akaike Information Criterion (AIC = 377.0511).  
*  We therefore adopt ARIMA(1,0,3) as the best model using 5% of significance level 
*****************************************************



*****************************************************
*  Step 3: Hyperparameter Tuning via Grid Search
*
*  We will conduct a systematic grid search over
*  ARIMA(p,0,q) combinations (e.g. p = 0…3, q = 0…3)
*  to confirm that ARIMA(1,0,3) is indeed the optimal
*  specification across a broader model space by
*  comparing AIC/BIC and diagnostic statistics.
*****************************************************


arimasoc pi, maxar(3) maxma(3) 
arima pi, arima(1,0,3)

*****************************************************
*  Conclution
*
*  Executing:
*    arimasoc pi, maxar(3) maxma(3)
*
*  confirms that ARIMA(1,0,3) remains the superior model
*  according to both AIC and BIC across all ARIMA(p,0,q)
*  configurations with p, q ≤ 3.
*****************************************************


*****************************************************
*  Step 6: Residual Normality & Unit Root Testing
*
*  Having selected ARIMA(1,0,3) as our final model, we now:
*    1. Assess the residuals for normality (e.g., Jarque–Bera, Shapiro–Wilk)
*    2. Confirm the absence of a unit root in the residuals
*       via the Augmented Dickey–Fuller test
*
*  These diagnostics ensure that our model errors behave
*  as white noise and that no remaining non-stationarity
*  biases our forecasts.
*****************************************************

* 1. Generate residual series
predict Errores103, resid

* 2. Plot inverse AR & MA roots (all should lie inside unit circle)
estat aroots

* 3. Bartlett's (B) test via cumulative periodogram for white noise
wntestb Errores103

* 4. Ljung-Box Q test up to lag 40 for any remaining autocorrelation
wntestq Errores103, lags(40)



 
*****************************************************
*  Conclution
*
*  1. ARMA Polynomial Roots:
*     - Plotting inverse AR and MA roots shows all points
*       strictly inside the unit circle, confirming stability
*       and invertibility of ARIMA(1,0,3).
*
*  2. Residual White-Noise Tests:
*     - Cumulative periodogram (Bartlett's B = 0.48, p = 0.9771)
*     - Portmanteau Q statistic = 19.8471 (df = 40, p = 0.9968)
*       Both tests fail to reject the null of white noise,
*       indicating no remaining autocorrelation.
*
*  Conclusion:
*    ARIMA(1,0,3) is stable and its residuals are white noise,
*    making it a reliable model for forecasting Maryland inflation.
*****************************************************
 
 
 

 
 
*****************************************************
*  ARIMA Model Selection at 1% Significance Level
*
*  Identifying the Optimal ARIMA(p,0,q) Specification
*  Using a 1% Parameter Inclusion Criterion
*****************************************************

* ADF test on the first‐difference
dfuller d.pi

*****************************************************
*  Conclution
*
*  ADF Test on Differenced Inflation Series
*
*  By taking the first difference Δπₜ = πₜ – πₜ₋₁, we remove any
*  unit‐root behavior, rendering the series stationary by design.
*  We now verify stationarity formally:
*    – H₀: Δπₜ has a unit root  
*    – H₁: Δπₜ is stationary  
*****************************************************

*****************************************************
*  Step 1: Model Identification via ACF & PACF
*****************************************************

ac d.pi, name (pi2)
pac d.pi, name (pii2)
graph combine pi2 pii2, r(2) name(piii2)

*****************************************************
*  Conclution

*  ACF/PACF Analysis & Candidate ARIMA(p,1,q) Models
*
*  After first‐differencing Δπₜ, the ACF shows a pronounced
*  negative spike at lag 1 and rapid decay within the 95% bands,
*  while the PACF exhibits significant cuts at lags 1 and 4.
*  This pattern indicates both autoregressive and moving-average
*  dynamics.  Accordingly, we will estimate and compare the
*  following ARIMA(p,1,q) specifications:
*    • ARIMA(1,1,1)
*    • ARIMA(1,1,4)
*    • ARIMA(4,1,1)
*    • ARIMA(4,1,4)
*****************************************************


*****************************************************
*  Step 2: Model Estimation & AIC Comparison
*****************************************************

arima pi, arima(1,1,1)
estat ic

arima pi, arima(1,1,4)
estat ic

arima pi, arima(4,1,1)
estat ic

arima pi, arima(4,1,4)
estat ic

*****************************************************
*  Conclution
*
*  Among the candidate ARIMA(p,1,q) specifications—
*    • ARIMA(1,1,1)
*    • ARIMA(1,1,4)
*    • ARIMA(4,1,1)
*    • ARIMA(4,1,4)
*  ARIMA(1,1,4) achieves the lowest AIC and is therefore
*  selected as the optimal model for Δπₜ.
*****************************************************



*****************************************************
*  Step 3: Hyperparameter Tuning via Grid Search
*****************************************************

arimasoc d.pi, maxar(4) maxma(4) 

*****************************************************
*  Conclution
*
*  Conducting a comprehensive grid search over ARMA(p,q)
*  for the differenced series confirms that ARMA(1,4)
*  is the optimal specification—exactly as suggested
*  by the ACF and PACF diagnostic plots.
*****************************************************
arima pi, arima(1,1,4)


*****************************************************
*  Step 6: Residual Normality & Unit Root Testing
*****************************************************

* 1. Generate residual series
predict diferencia114, resid

* 2. Plot inverse AR & MA roots (all should lie inside unit circle)
estat aroots // There is one unit root, this model CANNOT be used for forecast

* 3. Bartlett's (B) test via cumulative periodogram for white noise
wntestb diferencia114

* 4. Ljung-Box Q test up to lag 40 for any remaining autocorrelation
wntestq diferencia114, lags(40)


*****************************************************
*  Conclution
*
*  Stability Check Failure: ARIMA(1,1,4)
*
*  A unit root is detected in the AR polynomial of ARIMA(1,1,4),
*  indicating non-invertibility and instability.
*  Consequently, this model is unsuitable for forecasting.
*  We will instead adopt the next-best candidate specification
*  to ensure reliable out-of-sample predictions.
*****************************************************



*****************************************************
*  Step 7: Find New Models
*****************************************************

*  Best model using Box-Jenkyns Methodology
arima pi, arima(4,1,1)
predict diferencia411, resid


wntestq diferencia411
wntestb diferencia411
estat aroots


*  Best model using gridsearch and AIC BIC
arima pi, arima(2,1,3)
predict diferencia213, resid


wntestq diferencia213
wntestb diferencia213
estat aroots

*****************************************************
*  Conclution
*
*  • Based on ACF/PACF diagnostics, ARIMA(4,1,1) emerges as
*    the top runner-up after ARIMA(1,1,4).
*  • A comprehensive grid search identifies ARIMA(2,1,3) as
*    the optimal alternative specification.
*  • Both ARIMA(4,1,1) and ARIMA(2,1,3) pass all residual
*    white-noise tests (Ljung–Box, Bartlett's) and stability
*    checks (inverse roots inside the unit circle).
*
*  We will therefore include both models in our forecast
*  evaluation to compare their out-of-sample performance.
*****************************************************




*****************************************************
*  Step 8: One-Year (4-Quarter) Dynamic Forecast
*
*  We extend the time series by four quarters (one year)
*  and generate dynamic inflation forecasts from 1998Q2
*  for each selected ARIMA model:
*    - arm103pronsostico (ARIMA(1,0,3))
*    - arm213pronsostico (ARIMA(2,1,3))
*    - arm411pronsostico (ARIMA(4,1,1))
*****************************************************

tsappend, add(4)
predict arm103pronsostico, y dynamic(q(1998q2))
predict arm213pronsostico, y dynamic(q(1998q2))
predict arm411pronsostico, y dynamic(q(1998q2))



** Made by Pablo Reyes ** 





