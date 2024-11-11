#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 22:57:11 2024

@author: emilia
"""

import os
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns


# set working directory -- this is where your data is located on your computer
os.chdir('/home/emilia/data/project/piotr_koslinski')

# Open data
df = pd.read_csv('database-multiple-sclerosis-myasthenia.csv',sep = '\t',header = 0)

## Rename columns
# google rename function in python and see what is the syntax

df.rename(columns={'ID Pacjenta': 'ID', 'postać': 'type', 'Czas trwania': 'Duration', 'wiek': 'age',  'Plec': 'sex', 'conc': 'LogC', 'Lek': 'drug', 'miejsce': 'place'}, inplace=True)


# Inspect columns, shape, size
df.shape
df.size
df.columns
df.head()
df.info()

# Do we have NANs in columns?

##### Descriptive statistics ####
## description of patients /controls ##

case = df.loc[df['status'] == 'case'] #160
control = df.loc[df['status'] == 'control'] #53


### Cases: Covariates characteristics for multiple sclerosis ##
case.groupby('type')['type'].count()
case.groupby('drug')['drug'].count()
case.groupby('place')['place'].count()

case.Duration.value_counts(dropna=False)
case.EDSS.value_counts(dropna=False)
case.sex.value_counts(dropna=False)

case[["age", "Duration", "EDSS"]].describe()# basic stat for numerical variables


### Controls: Covariates characteristics for multiple sclerosis ##
control.sex.value_counts(dropna=False)
control[["age"]].describe()

############################################################
## Basic statistics for categorical & numerical variables ##
############################################################

# Chcek whether there is a statistical significance between sex distribution between cases and controls
from scipy.stats import chi2_contingency
import scipy.stats as stats


tabulated_data = [[103, 20], [45,30]]
chi2, p, dof, expected = stats.chi2_contingency(tabulated_data)
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}") #significant association between sex and status of the diesease

# Chcek whether there is a statistical significance between age distribution between cases and controls
from scipy.stats import mannwhitneyu
from scipy.stats import shapiro # check normality of distribution

shapiro(case[['age']].dropna())# most of aa have non-normal distribution

ca = case['age']
co = control['age']
# t-test
t_stat, p_value = stats.ttest_ind(ca.dropna(), co.dropna(), equal_var=False)
print(f"P-value: {p_value}")


####################
## Concentrations ##
####################

## Standardize concentrations ##
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

#dta1  = dta1[~dta1.index.duplicated()]

scaler = StandardScaler()

# transform data
df_conc =  df.iloc[:, 9:38]
df_conc = scaler.fit_transform(df.iloc[:, 9:38])
df_conc = pd.DataFrame(df_conc)
df_conc.columns =['SER_conc', 'GLN_conc', 'ARG_conc', 'CIT_conc',
       'ASN_conc', '1MHIS_conc', '3MHIS_conc', 'HYP_conc', 'GLY_conc',
       'THR_conc', 'ALA_conc', 'GABA_conc', 'SAR_conc', 'BAIB_conc',
       'ABA_conc', 'ORN_conc', 'MET_conc', 'PRO_conc', 'LYS_conc', 'ASP_conc',
       'HIS_conc', 'VAL_conc', 'TRP_conc', 'AAA_conc', 'LEU_conc', 'PHE_conc',
       'ILE_conc', 'C-C_conc', 'TYR_conc']

df_conc.rename(columns={'SER_conc': 'SER', 'GLN_conc' : 'GLN', 'ARG_conc' : 'ARG', 'CIT_conc' : 'CIT', 'ASN_conc' : 'ASN', '1MHIS_conc' : '1MHIS', '3MHIS_conc' : '3MHIS', 'HYP_conc' : 'HYP', 'GLY_conc' : 'GLY', 'THR_conc': 'THR', 'ALA_conc': 'ALA', 'GABA_conc' : 'GABA', 'SAR_conc': 'SAR', 'BAIB_conc': 'BAIB', 'ABA_conc': 'ABA', 'ORN_conc': 'ORN', 'MET_conc': 'MET', 'PRO_conc' : 'PRO', 'LYS_conc': 'LYS', 'ASP_conc' : 'ASP', 'HIS_conc': 'HIS', 'VAL_conc' : 'VAL', 'TRP_conc': 'TRP', 'AAA_conc' : 'AAA', 'LEU_conc' : 'LEU', 'PHE_conc': 'PHE','ILE_conc': 'ILE', 'C-C_conc' :'C-C', 'TYR_conc': 'TYR'}, inplace=True)

cov = df.iloc[:, 0:9]
dta1 = cov.reset_index(drop=True).join(df_conc)

#merge 2 categories
dta1.replace(to_replace="general",
           value="GMG", inplace=True);
dta1.replace(to_replace="eye-type",
           value="OMG", inplace=True);

dta1.groupby('type')['type'].count()

#%%
# Are concentrations of 29 AA different between cases and controls?

import scipy.stats as stats

#variables = ['ILE_conc', 'C-C_conc', 'TYR_conc']  # Add your variables here
variables = case.columns[9:]
results = {}

for var in variables:
    stat, p_value = stats.mannwhitneyu(x=case[var].dropna(), y=control[var].dropna(), alternative='two-sided')
    results[var] = {'U statistic': stat, 'p-value': p_value}

# Print the results
for var, res in results.items():
    print(f"Variable: {var}, p-value: {res['p-value']}")

import seaborn as sns
import matplotlib.pyplot as plt

# Create the boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x='AA', y='conc', hue='status', data=xx)
plt.title('Comparison of Distributions for Multiple Variables')
plt.show()
#%%

## Melting data ##

# melt() transform a DataFrame from wide format to long format
# it "melts" the DataFrame by unpivoting one or more columns, turning columns into rows,  which can be useful for data analysis and visualization

# id_vars - columns which WILL NOT be melted, everything else will be pivoted to rows.

xx = dta1.melt(id_vars=["status", "ID", "type", "Duration", "EDSS", "age", "sex","drug", "place"],
        var_name="AA",
        value_name="conc")



## Basic figures describing AA panel ##

## Density plot of AA concentrations ##
g = sns.FacetGrid(xx, col="AA", hue="status", col_wrap=5,height=2.3,palette=["red", "blue"])
g = g.map(sns.kdeplot,"conc", cut=0, fill=True, common_norm=False, alpha=.5)
g.add_legend()
g.refline(x=xx["conc"].median())
g.set_titles(col_template="{col_name}", row_template="{row_name}")

g.savefig("/home/emilia/data/project/piotr_koslinski/figures_for_pub/Fig_1_python-class-density-plot.png", dpi=300, bbox_inches='tight')


# Boxplots of AA concentrations
sns.set(font_scale=1.0)
colors = {"case": "red", "control": "blue"}

g = sns.catplot(data=xx, x='status', y='conc', col='AA', kind='box', col_wrap=6,
                sharey=True, height=3.5, aspect=.6, dodge=False, palette=colors)
g.set_axis_labels("", "Concentration")
g.set_titles(col_template="{col_name}", row_template="{row_name}")


#Manually define p-values (arbitrary values for demonstration)
manual_p_values = [ 0.58, 0.015,0.0001, 0.0008, 0.602, 0.00002, 0.315, 0.02, 0.417, 0.566, 0.154, 0.202, 0.041, 0.193, 0.0008, 0.824,0.229, 0.0001, 0.233, 0.219, 0.899, 0.011, 0.0006, 0.779 ,0.08, 0.147, 0.96, 0.005, 0.516 ]

#Add p-value annotations
for i, ax in enumerate(g.axes.flat):
    ax.text(0.5, 0.95, f'p = {manual_p_values[i]:.3f}', transform=ax.transAxes,
            horizontalalignment='center', verticalalignment='top', fontsize=10)

plt.tight_layout()
g.fig.subplots_adjust(right=0.95)
g.savefig("/home/emilia/data/project/piotr_koslinski/figures_for_pub/Fig_2_python-class-boxplot-plot.png", dpi=300, bbox_inches='tight')
plt.show()



# check if AA concentration differ between disease type -- kruskal wallis test statistics ##

ppms = xx.loc[xx['type'] == 'PPMS'] # subset only PPMS cells
ppms_1= ppms[['conc']]
smps =  xx.loc[xx['type'] == 'SPMS']
smps_1 = smps[['conc']]

rrms =  xx.loc[xx['type'] == 'RRMS']
rrms_1 = rrms[['conc']]

gen = xx.loc[xx['type'] == 'GMG']
gen_1 = gen[['conc']]

ocular = xx.loc[xx['type'] == 'OMG']
ocular_1 = ocular[['conc']]


from scipy import stats
stats.kruskal(ppms_1['conc'].dropna() ,smps_1['conc'].dropna(), rrms_1['conc'].dropna() , gen_1['conc'].dropna()) # what doeas the p-value say?

# between which groups there are differences
xx_to_kruskal = pd.concat([ppms, smps,  rrms], axis=0)
import scikit_posthocs as sp
sp.posthoc_conover(xx_to_kruskal, val_col='EDSS', group_col='type', p_adjust = 'holm')



## Plot overall AA concentration stratified by disease type ##
%matplotlib qt
sns.set(font_scale= 1)
xx_to_kruskal = pd.concat([ppms, smps,  rrms, gen], axis=0)
g1 = sns.catplot(x="type", y="conc", hue="type", kind="box", data=xx_to_kruskal,height=8, aspect=1 );
g1.set(xlabel=None)
g1.set(ylabel='Concentration')
# Tighten layout
plt.tight_layout()
g1.set(xticklabels=["PPMS", "SPMS", "RRMS", "GMG"])
#g1.savefig("/home/emilia/data/project/piotr_koslinski/figures_for_pub/Fig_1.png", dpi=300, bbox_inches='tight')


# Is there any linear relationship with age
sns.set(font_scale= 1.5)
g = sns.lmplot(col="AA",x="age",y="conc", palette="Set2",x_jitter=1.3,
               data=xx, col_wrap=5, height=2.5, legend=False,scatter_kws={'color': 'grey'})
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.set_axis_labels( "age")

# density plot for concentrations
g=sns.displot(xx, x='conc', hue='status', col='AA', kind='kde', col_wrap=6, fill=True,
              height=2.5, aspect=2.9)
g.set_axis_labels("", "conc")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
# Adjust x-scale of subplots
for ax in g.axes.flat:
    ax.set_xlim(-2, 20)  # Set x-axis limits here
plt.show()

# concentration between sexes
sns.set(font_scale=1.0)
colors = {"Male": "blue", "Female": "pink"}

g = sns.catplot(data=xx, x='sex', y='conc',hue='sex', col='AA', kind='box', col_wrap=6,
                sharey=True, height=3.5, aspect=.6, dodge=False, palette=colors)
g.set_axis_labels("", "Concentration")
g.set_titles(col_template="{col_name}", row_template="{row_name}")


# concentration between disease types
g=sns.catplot(data=xx, x='type', y='conc',  col='AA',  kind='box', col_wrap=6,
              sharey=True, height=2.5, dodge=False)
g.set_axis_labels("", "conc")
g.set_titles(col_template="{col_name}", row_template="{row_name}")