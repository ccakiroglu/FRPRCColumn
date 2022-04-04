from pandas import read_csv
from numpy import array
homedir='G:\\My Drive\\Papers\\2022\\FRPRCColumn\\EXCELCSV\\FRP-RC_Columns_Database_Concentric_OneHot.csv'
#sutunlar = [r'$L$',r'$\\gamma$',r'$\\phi$',r'$q$',r'$Cost$', r'$D$']
sutunlar = [r'$\lambda$',r'A$_g$',r'Circular',r'NWC',r'LWC',r'f$_c$', r'GFRP$_L$', r'CFRP$_L$', r'$\rho$', r'E$_L$', r'fu$_L$',\
        r'GFRP$_H$', r'CFRP$_H$', r'steel$_H$', r'spiral', r'ties', r'spacing$_H$', r'P$_exp$']
#colnames = [r'$L$',r'$\gamma$',r'$\phi$',r'$q$',r'$Cost$', r'$D$']
vericercevesi = read_csv(homedir,header=0, names=sutunlar)
#vericercevesi = read_csv(homedir, names=sutunlar)
dizimiz = vericercevesi.values
X = dizimiz[:,0:5]
y = dizimiz[:,5]

X=vericercevesi[[r'$\lambda$',r'A$_g$',r'Circular',r'NWC',r'LWC',r'f$_c$', r'GFRP$_L$', r'CFRP$_L$', r'$\rho$', r'E$_L$', r'fu$_L$',\
        r'GFRP$_H$', r'CFRP$_H$', r'steel$_H$', r'spiral', r'ties', r'spacing$_H$']]
#The slashes (escape characters) above cause the problem
y=vericercevesi[[r'P$_exp$']]
import xgboost
import shap
from lightgbm.sklearn import LGBMRegressor
from matplotlib import pyplot
pyplot.rc('text',usetex=True)
pyplot.rc('font',size=25)
pyplot.rcParams.update({'font.size': 25})
# train an XGBoost model
model = xgboost.XGBRegressor().fit(X, y)
#model = LGBMRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
#explainer = shap.Explainer(model)
explainer = shap.TreeExplainer(model,X)
shap_values = explainer(X)

#shap.initjs()
#shap.plots.force(shap_values[0], matplotlib=True, show=False)
pyplot.xlabel(r'$SHAP\hspace{0.5em} value$')
#pyplot.xlabel(r'$b$')
#pyplot.ylabel(r'$SHAP \hspace{0.5em}value\hspace{0.5em} for$')
#shap.plots.beeswarm(shap_values, show=False,color_bar_label=r'$Feature\hspace{0.5em} value$' )
pyplot.tight_layout()
shap.plots.beeswarm(shap_values)
pyplot.tight_layout()
#shap.plots.beeswarm(shap_values,color_bar_label=r'$Feature\hspace{0.5em} value}$')
#shap.plots.scatter(shap_values[:,r'$h$'],show=False, color=shap_values)#color_bar_labels
#fig, ax = pyplot.gcf(), pyplot.gca()
#ax.set_xlabel(r'$SHAP\hspace{0.5em} value$', fontdict={"size":15})
#ax.set_ylabel(r'$SHAP \hspace{0.5em}value\hspace{0.5em} for\hspace{0.5em}h$', fontdict={"size":25})
#ax.set_yticks(array([-300,-200,-100,0,100,200, 300, 400,500]))
#ax.set_xticks(array([0,250,500,750,1000,1250,1500,1750,2000]))
#ax.tick_params(axis='y', labelsize=25)
#ax.tick_params(axis='x', labelsize=25)
#ax.set_xlabel(r'$h$',fontdict={"size":25})
#pyplot.savefig('/mnt/g/My Drive/Papers/2022/CylindricalWall/IMAGES/LightGBMSHAP.svg')
#pyplot.savefig('G:\\My Drive\\Papers\\2022\\CylindricalWall\\IMAGES\\LightGBMSHAP.svg')
