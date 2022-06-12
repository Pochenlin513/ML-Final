In Preprocessing.py, we have the following functions, with lists of numerical features and categorical features.
See how to run these functions in Preprocessing.ipynb

Function 1 will deal w/ missing data:

1. RemoveData(ds, ds_test, cols=['最高學歷', '畢業學校類別']): removes
    (1) 最高學歷 and 畢業學校類別 that have many missing data, for both training & testing data
    (2) 73 persons that have no data in training
    
Function 2~6 are used for generating features:

2. seasonAdd(ds_season): adds cols in season.csv to create more feat
3. subData(ds_season, year, Q): returns season data for certain year and period, add period to cols' name
4. subDataYear(ds_season, year): for each year, each person, merge data of different period
5. mergeSeason(ds_train, ds_test, ds_season): added season data into training and testing set
6. TrainAdd(ds_train): add those feat w/ A, B, and probably C to form a new feat

Function 7, 8, perform data normalization for numerical features

7. stdNormalize(df_train, df_test, NUM_FEATS): normalizes by 
    (data - mean) / std
8. maxminNormalize(df_train, df_test, NUM_FEATS): normalizes by
    (data - min) / (max - min)

9. splitXY(ds): splits raw data into features & labels

Function 10~14 perform feature selection:

10. keepMutual(X, Y, k=10): chooses k-best feat according to mutual info
11. keepChi(X, Y, k=10): chooses k-best feat according to Chi-squared Score
12. keepANOVA(X, Y, k=10): chooses k-best feat according to ANOVA
13. keepNMF(x_train, x_test, k): reduce feat to k-dim using NMF
14. keepPCA(x_train, x_test, k): reduce feat to k-dim using PCA