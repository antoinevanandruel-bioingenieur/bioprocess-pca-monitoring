# Multivariated Statistical Process Control (MSPC) for Bioprocess Monitoring

## Executive Summary
In biopharmaceutical manufacturing (MSAT), living cell cultures (e.g., mammalian cells producing monoclonal antibodies) exhibit high non-linear variability. Monitoring individual process parameters (pH, Temperature, dissolved Oxygen) isolated from one another leads to late deviation detection. 

This project implements an **unsupervised mathematical model using Principal Component Analysis (PCA)** to monitor a bioreactor multivariate space in real-time, computing mathematical limits to trigger early alarms before a batch failure occurs.

## The Mathematical Framework

The model transforms highly correlated multi-sensor data into a lower-dimensional orthogonal space. Two complementary statistical metrics are tracked:

### 1. Hotelling's $T^2$ Statistic (In-Model Distance)
It measures the variation within the PCA latent space (the main principal components). It detects shifts along the normal direction of the process.

$$T^2 = x^T P_a \Lambda^{-1} P_a^T x$$

Where:
* $x$ is the standardized measurement vector.
* $P_a$ is the matrix of the $a$ retained loadings (eigenvectors).
* $\Lambda^{-1}$ is the inverse diagonal matrix of the eigenvalues.

The statistical control limit is established using the **Fisher-Snedecor $F$-distribution**:
$$T^2_{\alpha} = \frac{a(n^2 - 1)}{n(n - a)} F_{\alpha}(a, n - a)$$

### 2. Squared Prediction Error / $Q$-Residual (Out-of-Model Distance)
It measures the perpendicular distance from the sample to the PCA pool. It detects when a new correlation break occurs (e.g., a novel unmodeled failure mode).

$$Q = e^T e = ||x - \hat{x}||^2$$

The critical limit is calculated using the **Jackson-Mudholkar approximation** based on the eigenvalues of the residual variance matrix ($\theta_i$).

## Industrial Applicability (MSAT & QA alignment)
* **Early Warning Systems**: Detecting process drifts 12 to 24 hours before standard univariate threshold breaches.
* **Root Cause Analysis**: Projecting the residuals back to understand which specific sensor (e.g., a drifting pH probe) caused the statistical deviation.
