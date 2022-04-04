from pandas import read_csv, DataFrame, get_dummies
from numpy import absolute,arange,mean,std,argsort,sqrt
from numpy.random import ranf
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor 
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
#from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from lightgbm.sklearn import LGBMRegressor
from matplotlib import pyplot 
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score, accuracy_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
pyplot.rc('text',usetex=True)
pyplot.rc('font',size=15)
#pyplot.figure(figsize=(6.4,4.26666666667))
scaler = MinMaxScaler()
pyplot.rc('text',usetex=True)
pyplot.rc('font',size=15)
colnames = [r'Slenderness',r'Ag',r'Circular',r'NWC',r'LWC','fc', 'GFRP_L', 'CFRP_L', 'Rho', 'E_L', 'fu_L',\
        'GFRP_H', 'CFRP_H', 'steel_H', 'spiral', 'ties', 'spacing_H', 'Pexp']
homedir='G:\\My Drive\\Papers\\2022\\FRPRCColumn\\EXCELCSV\\FRP-RC_Columns_Database_Concentric_OneHot.csv'
df = read_csv(homedir,header=0,names=colnames)
data=df.values
# split into inputs and outputs
#X, y = df.iloc[:, :-1], df.iloc[:, -1]
X, y = data[:, :-1], data[:, -1]
print('X.shape:', X.shape,'y.shape', y.shape)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#XGBModel = XGBRegressor()
#GBMmodel=GradientBoostingRegressor()
#RandomForestModel = RandomForestRegressor()
#lassoModel=Lasso()
#AdaBoostModel = AdaBoostRegressor()
#LGBMmodel = LGBMRegressor()
#CatBoostModel = CatBoostRegressor()
#kernelRidgeModel=KernelRidge()
svrModel=SVR()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++
cv = KFold(n_splits=10, shuffle=True, random_state=1)
#Here we list the parameters we want to tune
space = dict()
#space['n_estimators'] = arange(1,600,50)
#space['n_estimators'] = [1, 20, 100, 500, 1000]
#space['max_depth']=[2, 5, 10, 20]
#space['min_child_weight']=[1, 5, 10]
#space['learning_rate']=arange(0.1, 4, 0.5)
#space['learning_rate']=[.03, 0.05, .07]
#space['learning_rate']=[0.02, 0.05, 0.1, 0.2]
#space['learning_rate']=[0.2]
#search = GridSearchCV(CatBoostModel, space, n_jobs=-1, cv=cv, refit=True)
#search = GridSearchCV(LGBMmodel, space, n_jobs=-1, cv=cv, refit=True)
#search = GridSearchCV(XGBModel, space, n_jobs=-1, cv=cv, refit=True)
#search = GridSearchCV(GBMmodel, space, n_jobs=-1, cv=cv, refit=True)
#search = GridSearchCV(RandomForestModel, space, n_jobs=-1, cv=cv, refit=True)
search = GridSearchCV(svrModel, space, n_jobs=-1, cv=cv, refit=True)
#search = GridSearchCV(kernelRidgeModel, space, n_jobs=-1, cv=cv, refit=True)
#search = GridSearchCV(lassoModel, space, n_jobs=-1, cv=cv, refit=True)
#search = GridSearchCV(AdaBoostModel, space, n_jobs=-1, cv=cv, refit=True)
result = search.fit(X_train, y_train)
best_model = result.best_estimator_
yhat_test = best_model.predict(X_test)
yhat_train = best_model.predict(X_train)
#pyplot.scatter(yhat_train,y_train, marker='s',facecolor='seagreen',edgecolor='seagreen', label=r'$Training\hspace{0.5em}set$')#original
#pyplot.scatter(yhat_test,y_test, marker='o',facecolor='forestgreen',edgecolor='forestgreen', label=r'$Random\hspace{0.5em}Forest \hspace{0.5em}K_s=300$')
#pyplot.scatter(yhat_test,y_test, marker='o',facecolor='lightseagreen',edgecolor='lightseagreen', label=r'$LightGBM\hspace{0.5em}K_s=300\hspace{0.5em}$')
pyplot.scatter(yhat_train,y_train, marker='^',facecolor='lightseagreen',edgecolor='lightseagreen', label=r'$Training\hspace{0.5em}set$')
pyplot.scatter(yhat_test,y_test, marker='s',facecolor='darkgoldenrod',edgecolor='darkgoldenrod', label=r'$Test\hspace{0.5em}set$')
#pyplot.scatter(yhat_test,y_test, marker='o',facecolor='fuchsia',edgecolor='fuchsia', label=r'$CatBoost\hspace{0.5em}K_s=300\hspace{0.5em}$')
#x=arange(25,625,25)
#x=arange(1000,2025,25)
#pyplot.plot(x,r2scoresLR02, color='blue', label=r'$L_r=0.02$')
#pyplot.plot(x,r2scoresLR05, color='red', dashes=[5,2,2,2], label=r'$L_r=0.05$')
#pyplot.plot(x,r2scoresLR1, color='green', dashes=[5,2], label=r'$L_r=0.1$')
#pyplot.plot(x,r2scoresLR2, color='yellow', dashes=[2,2], label=r'$L_r=0.2$')
#pyplot.xticks(arange(1000,2100,100))
#pyplot.title(r'$max\_depth=5$')
xk=[0,16000];yk=[0,16000];ykPlus10Perc=[0,17600];ykMinus10Perc=[0,14400];
pyplot.plot(xk,yk, color='black')
pyplot.plot(xk,ykPlus10Perc, dashes=[2,2], color='fuchsia')
pyplot.plot(xk,ykMinus10Perc,dashes=[2,2], color='fuchsia')
pyplot.grid(True)
pyplot.xlabel(r'$P_{predicted}\hspace{0.5em}[m]$')
pyplot.ylabel(r'$P_{exp}\hspace{0.5em}[kN]$')
pyplot.legend()
pyplot.tight_layout()
#pyplot.savefig('G:\\My Drive\\Papers\\2022\\CantileverSoldierPile\\IMAGES\\CatBoost5v300.svg')
pyplot.show()
#R2=r2_score(y_test, yhat)#original
print('MAPE train= ',mean_absolute_percentage_error(y_train, yhat_train))
print('RMSE train= ',sqrt(mean_squared_error(y_train, yhat_train)))
print('MAE train= ',mean_absolute_error(y_train, yhat_train))
print('R2 train:',r2_score(y_train, yhat_train))
print('MAPE test= ',mean_absolute_percentage_error(y_test, yhat_test))
print('RMSE test= ',sqrt(mean_squared_error(y_test, yhat_test)))
print('MAE test= ',mean_absolute_error(y_test, yhat_test))
print('R2 test:',r2_score(y_test, yhat_test))#original
print('Best parameters are',search.best_params_)#original
#f=open('y_test_CatBoost200.csv','w')
#for i in y_test:
#    f.write(str(i)+'\n')
#f.close()
#g=open('yhat_test_CatBoost200.csv','w')
#for j in yhat_test:
#    g.write(str(j)+'\n')
#g.close()
#y_test_short=[]
#yhat_test_short=[]
#for i in range(0,len(y_test),10):
#    y_test_short.append(y_test[i])
#    yhat_test_short.append(yhat_test[i])
#f=open('y_test_CatBoost200short.csv','w')
#for i in y_test_short:
#    f.write(str(i)+'\n')
#f.close()
#g=open('yhat_test_CatBoost200short.csv','w')
#for j in yhat_test_short:
#    g.write(str(j)+'\n')
#g.close()
#y_test_short=read_csv('G:\\My Drive\\Papers\\2022\\CantileverSoldierPile\\EXCELCSV\\y_test_CatBoost200short.csv').values
#yhat_test_short=read_csv('G:\\My Drive\\Papers\\2022\\CantileverSoldierPile\\EXCELCSV\\yhat_test_CatBoost200short.csv').values
#pyplot.figure(figsize=(12,4))
#pyplot.plot(y_test_short[0:500], marker='s',color='yellowgreen',alpha=1.0, label=r'$Actual$')
#pyplot.plot(yhat_test_short[0:500], marker='o',color='royalblue', alpha=1.0, label=r'$Predicted$')
#pyplot.legend()
#pyplot.xlabel(r"$Number\hspace{0.5em} of \hspace{0.5em}combinations$")
#pyplot.ylabel(r"$D[m]$")
#pyplot.grid(True)
#pyplot.tight_layout()
#pyplot.show()
