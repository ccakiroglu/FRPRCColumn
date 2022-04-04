from pandas import read_csv
from numpy import array
from xgboost.sklearn import XGBRegressor 
#homedir='G:\\My Drive\\Papers\\2022\\CantileverSoldierPile\\EXCELCSV\\SoldierPile5varsKs300.csv'
homedir='G:\\My Drive\\Papers\\2022\\FRPRCColumn\\EXCELCSV\\FRP-RC_Columns_Database_Concentric_OneHot.csv'
#sutunlar = [r'$L$',r'$\gamma$',r'$\phi$',r'$q$',r'$Cost$', r'$D$']
#colnames = [r'$L$',r'$\gamma$',r'$\phi$',r'$q$',r'$Cost$', r'$D$']
colnames = [r'Slenderness',r'Ag',r'Circular',r'NWC',r'LWC',r'fc', r'GFRP$_L$', r'CFRP$_L$', r'Rho', r'E$_L$', r'fu$_L$',\
        r'GFRP$_H$', r'CFRP$_H$', r'steel$_H$', r'spiral', r'ties', r'spacingH', r'Pexp']
vericercevesi = read_csv(homedir,header=0, names=colnames)
dizimiz = vericercevesi.values
X = dizimiz[:,0:17]
y = dizimiz[:,17]

X=vericercevesi[[r'Slenderness',r'Ag',r'Circular',r'NWC',r'LWC',r'fc', r'GFRP$_L$', r'CFRP$_L$', r'Rho', r'E$_L$', r'fu$_L$',\
        r'GFRP$_H$', r'CFRP$_H$', r'steel$_H$', r'spiral', r'ties', r'spacingH']]
y=vericercevesi[['Pexp']]
import xgboost
import shap
from catboost import CatBoostRegressor
from matplotlib import pyplot
pyplot.rc('text',usetex=True)
pyplot.rc('font',size=25)
pyplot.rcParams.update({'font.size': 25})
# train an XGBoost model
#model = CatBoostRegressor().fit(X, y)
model=XGBRegressor().fit(X,y)
# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.TreeExplainer(model)
#explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X)

#shap.initjs()
#shap.plots.force(shap_values[0], matplotlib=True, show=False)
#pyplot.xlabel('')
#pyplot.xlabel(r'$b$')
#pyplot.ylabel(r'$SHAP \hspace{0.5em}value\hspace{0.5em} for$')
pyplot.tight_layout()
#shap.plots.beeswarm(shap_values, show=False,color_bar_label=r'$Feature\hspace{0.5em} value$' )
#shap.summary_plot(shap_values,X)
#shap.plots.beeswarm(shap_values)
#shap.plots.scatter(shap_values[:,r'$r$'],show=False, color=shap_values)#color_bar_labels
#shap.plots.scatter(shap_values[:,1],show=False, color=shap_values)#color_bar_labels
fig, ax = pyplot.gcf(), pyplot.gca()
ax.set_xlabel(r'spacingH', fontdict={"size":15})
ax.set_ylabel(r'$SHAP \hspace{0.5em}value\hspace{0.5em} for\hspace{0.5em}spacingH$', fontdict={"size":15})
pyplot.tight_layout()
shap.dependence_plot(16, shap_values, X, ax=ax)
#shap.dependence_plot(r'$r$', shap_values, X, interaction_index=None)
#ax.set_yticks(array([-300,-200,-100,0,100,200, 300, 400,500]))
#ax.set_xticks(array([0,250,500,750,1000,1250,1500,1750,2000]))
#ax.tick_params(axis='y', labelsize=25)
#ax.tick_params(axis='x', labelsize=25)
#ax.set_xlabel(r'$h$',fontdict={"size":25})
#pyplot.savefig('G:\\My Drive\\Papers\\2022\\CylindricalWall\\IMAGES\\CatBoostSHAPfeatDepR.svg')
#print(shap_values.shape)
