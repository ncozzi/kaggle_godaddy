# Kaggle GoDaddy: Microbusiness density forecasting
*By @ncozzi*

**Link to competition:** https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting

---------------------------

"*A microbusiness is a type of small business that has under 10 employees (according to the SBA), meets specific annual revenue criteria set by state authorities, and has small startup needs.*" Source: https://www.thebalancemoney.com/what-is-a-microbusiness-5210249

In this competition, the goal is to forecast microbusiness density for 3135 counties in USA, with the goal of **minimum SMAPE** score for three, four and five months in advance - with the last available observation being December 2022. It involves longitudinal data, thus the techniques illustrated fit this scenario.

The following strategies are adopted:
- Focus on predicting MBD *growth* instead
- Minimum MAE of the growth per county
- Adjust the CV strategy according to the challenge's goal
- Use lagged values and rolling mean according to time series analysis results


Future extensions could use model ensembles; since this was created during my free time, no complex algorithms were used - a Linear SVR was selected

Hope this is helpful for any future analyses you perform!
