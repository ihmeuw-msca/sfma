import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pystoned import CNLS,StoNED
from pystoned.plot import plot2d
from pystoned.constant import CET_ADDI, FUN_PROD, RTS_VRS,RED_MOM,RED_QLE,RED_KDE
from pystoned.constant import CET_MULT, FUN_COST, RTS_VRS, RED_MOM,RED_QLE,RED_KDE
from pystoned import CNLS, StoNED
from scipy import optimize
from pystoned.plot import plot2d
from pystoned.constant import CET_ADDI, FUN_PROD, RTS_VRS,RED_MOM,RED_QLE,RED_KDE
from sfma import Data, SFMAModel, Variable, SplineVariable, SplineGetter, SplinePriorGetter, UniformPrior
class comp_sfa:
    draws = 200
    scale = 0.01
    u_scale = 0.1
    x = np.arange(1,200)
    u = np.arange(1,200)
    df = pd.DataFrame([x,u])
    def generate_(self,type,n_sample,e_scale, u_scale):
        x = np.sort(np.random.uniform(low=0, high=1, size=n_sample)) # 
        u = np.abs(np.random.normal(loc=0, scale=u_scale, size=n_sample)) # (U_scale)
        if (type == "homosk"): # homoskedastic
            se = e_scale
            eps = np.random.normal(loc=0, scale=se, size=n_sample)
            
        elif (type == "heterosk"): # Heteroskedastic and increasing variance scale with X 
#             se = (x*scale)
            se = np.sqrt(x*e_scale)
            eps =  np.random.normal(loc=0, scale = se, size=n_sample)
            
        else:
            print('Invalid !!') 
        y_true = 3 + np.log(x)
        y = y_true + eps - u
        val_gen = pd.DataFrame({
            'x':x,
            'y':y,
            'se':se,
            'y_true':y_true})
        return (pd.DataFrame(val_gen)) 
    def stoned_(self,df): 
        x = df.iloc[:,0]
        y = df.iloc[:,1]
        # define the CNLS model
        model = CNLS.CNLS(y, x, z=None, cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
        model.optimize('nahomw@uw.edu')
        ########################################################
         # define the CNLS model
        xx = np.array(model.get_frontier()).T
        xx = np.array(xx).T
        rd = StoNED.StoNED(model)
        xy = np.array(StoNED.StoNED.get_frontier(rd,method=RED_QLE)).T
        xz = np.array(StoNED.StoNED.get_frontier(rd,method=RED_MOM)).T
#         zz = np.array(StoNED.StoNED.get_frontier(rd,method=RED_KDE)).T 
        ##########################################################
        y_cnls = pd.DataFrame(xx)
        y_qle = pd.DataFrame(xy)
        y_mom = pd.DataFrame(xz)
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        val_mod = pd.concat([x,y,y_cnls,y_qle,y_mom],axis = 1)
        val_mod.columns = ['x','y','y_cnls','y_qle','y_mom']
        return (val_mod)

    def sfma_(self,df):
        x = df['x']
        y = df['y']
        se = df['se']
        x_var = "x"
        y_var = "y"
        data = Data(
            obs=y_var,
            obs_se="se"
        )

        priors = [
            SplinePriorGetter(UniformPrior(lb=0.0, ub=np.inf), order=1, size=20),
            SplinePriorGetter(UniformPrior(lb=-np.inf, ub=0.0), order=2, size=20)
                ]
        variables = [
            SplineVariable(x_var,
                        spline=SplineGetter(knots=np.linspace(0.0, 1.0, 7),
                                            degree=3,
                                            r_linear=False,
                                            l_linear=False,
                                            knots_type="rel_domain",
                                            include_first_basis=True),
                        priors=priors)
        ]

        model = SFMAModel(data, variables)
        model.attach(df)

        model.parameter.variables[0].spline.design_dmat(np.array([12]), order=2)

        model.parameter.variables[0].spline.knots

        model.fit(outlier_pct=0.0, trim_verbose=True, trim_max_iter=15, trim_step_size=2.0,
                eta_options={"method": "bounded", "bounds": [0.0, 1.0]})

        df_pred = pd.DataFrame({
#             x_var: np.linspace(df[x_var].min(), df[x_var].max(), len(df))
            x_var: df['x']
            })
        df_pred["pred"] = model.predict(df_pred)
        return (df_pred)
    
