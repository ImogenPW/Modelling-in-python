#Author - Imogen Poyntz-Wright
#Date - 13/06/24
#Title - Learning how to run linear models, general linear models and generalised linear models

#Load libraries
import pandas as pd
import numpy as np
import seaborne as sn
import statsmodels.api as sm
import statsmodels.formula.api as smf


#Load dataframes
data = pd.read_excel("english_education.xlsx")


#Clean data 
##Remove NAs
data_cleaned = data.dropna()


#Linear Model
### For noramally distributed residuals and linear relationships
print(data_cleaned.head())
X = data_cleaned['population_2011'] #independent variable
y = data_cleaned['education_score'] #dependent variable
X = sm.add_constant(X) #add constant

model_lm = sm.OLS(y, X).fit()
print(model_lm.summary())

plt.clf()
model_lm_residuals = model_lm.resid
plt.scatter(model_lm.fittedvalues, model_lm_residuals)
plt.title('Residuals vs. Predicted Values')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()


#General Linear Model
### For non-normally distrubted residuals or count data etc and non-linear relationships
X = data_cleaned['population_2011'] #independent variable
y = data_cleaned['education_score'] #dependent variable 
X = sm.add_constant(X) #add constant

model_glm = sm.GLM(y, X, data_cleaned, family = sm.families.Guassian()).fit()
print(model_glm.summary())

plt.clf()
model_glm_residuals = model_glm.resid_deviance
plt.scatter(model_glm.fittedvalues, model_glm_residuals)
plt.title('Residuals vs. Predicted Values')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()


#Generalised Linear Mixed Model
### For non-normally distrubted residuals or count data etc and non-linear relationships AND random effects/nested variables
model_glmm = smf.mixedlm('education_score ~ population_2011 + C(rgn11nm)', data_cleaned, groups = data_cleaned['coastal']).fit()
print(model_glmm.summary())

plt.clf()
residuals = model_glmm.resid
plt.scatter(model_glmm.fittedvalues, residuals)
plt.title('Residuals vs. Predicted Values')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()




#Plot graph
sns.scatterplot(data = data_cleaned, x= 'education_score', y = 'population_2011', hue = 'rgn11nm')
plt.xlabel = 'Education Score'
plt.ylabel = 'Population Size'
plt.show()

