# Homework 1

## Problem 4: Normality of regression residuals (ALE dataset)

```bash
cd homework_1
pip install -r requirements.txt
python problem4.py
```

The script reads `data/ale.csv`, which is included in this repo. (The dataset is not available
through the `ucimlrepo` package, so it was downloaded manually from
https://archive.ics.uci.edu/dataset/844.)

Outputs:
- `figures/qqplots.png`: QQ plots of residuals for each model (top row: out-of-fold, bottom row: training)
- `results.csv`: RMSE, `scipy.stats.normaltest` statistic and p-value, skewness, and excess kurtosis

Data: Singh, A., Kotiyal, V., Sharma, S., Nagar, J., & Lee, C. (2020). *Average Localization Error (ALE)
in sensor node localization process in WSNs* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5ZP6R (CC BY 4.0)
