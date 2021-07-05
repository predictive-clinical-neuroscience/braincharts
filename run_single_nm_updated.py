import os
import sys
import glob
import numpy as np
import pandas as pd
import pickle
import shutil
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy import optimize
import seaborn as sns
sns.set(style='whitegrid')

from pcntoolkit.normative import estimate, predict, evaluate
from pcntoolkit.util.utils import create_bspline_basis, compute_MSLL, create_design_matrix
from nm_utils import save_output, test_func, calibration_descriptives

###################################CONFIG #####################################

data_dir = '/Users/andmar/data/sairut/data'
out_dir = '/Users/andmar/data/sairut/results'

df_tr = pd.read_csv(os.path.join(data_dir,'lifespan_big_controls_tr.csv'), index_col=0) 
df_te = pd.read_csv(os.path.join(data_dir,'lifespan_big_controls_te.csv'), index_col=0)

# remove some bad subjects
df_tr = df_tr.loc[df_tr['EstimatedTotalIntraCranialVol'] > 0.5]
df_te = df_te.loc[df_te['EstimatedTotalIntraCranialVol'] > 0.5]

cols_cov = ['age','sex']
#cols_site = df_te.columns.to_list()[222:261]
cols_site = df_te.columns.to_list()[225:304]
cols = cols_cov + cols_site

# # configure IDPs to use
# idp_ids_lh = df_te.columns.to_list()[15:90]
# idp_ids_rh = df_te.columns.to_list()[90:165]
# idp_ids_sc = df_te.columns.to_list()[165:198]
# idp_ids_glob = df_te.columns.to_list()[208:216] + df_te.columns.to_list()[220:222]
# idp_ids_all = df_te.columns.to_list()[15:207] + df_te.columns.to_list()[208:222]
# #idp_ids = ['lh_S_temporal_sup_thickness']
# idp_ids = ['Left-Lateral-Ventricle']
# idp_ids = ['rh_S_suborbital_thickness']
# idp_ids = ['lh_S_orbital_med-olfact_thickness']
# #idp_ids = idp_ids_lh + idp_ids_rh + idp_ids_sc + idp_ids_glob

#idp_ids_lh = df_te.columns.to_list()[4:79]
idp_ids_lh = df_te.columns.to_list()[3:78]
idp_ids_rh = df_te.columns.to_list()[80:155]
idp_ids_sc = df_te.columns.to_list()[155:187]
idp_ids_glob = ['SubCortGrayVol', 'TotalGrayVol', 'SupraTentorialVol', 
                'SupraTentorialVolNotVent','BrainSegVol-to-eTIV', 'MaskVol-to-eTIV',
                'avg_thickness', 'lhCerebralWhiteMatterVol', 'rhCerebralWhiteMatterVol']
idp_ids = idp_ids_lh + idp_ids_rh + idp_ids_sc + idp_ids_glob

# run switches
show_plot = True
force_refit = True
outlier_thresh = 7

cov_type = 'bspline'  # 'int', 'bspline' or None
warp =  'WarpSinArcsinh'   # 'WarpBoxCox', 'WarpSinArcsinh'  or None
sex = 0 # 1 = male 0 = female
if sex == 1: 
    clr = 'blue';
    sex_name = 'male'
else:
    clr = 'red'
    sex_name = 'female'

# limits for cubic B-spline basis 
xmin = -5 # boundaries for ages of UKB participants +/- 5
xmax = 110

################################### RUN #######################################
# create dummy data for visualisation
xx = np.arange(xmin,xmax,0.5)
if len(cols_cov) == 1:
    print('fitting sex specific model')
    X0_dummy = np.zeros((len(xx), 1))
    X0_dummy[:,0] = xx
    df_tr = df_tr.loc[df_tr['sex'] == sex]
    df_te = df_te.loc[df_te['sex'] == sex]
else:
    X0_dummy = np.zeros((len(xx), 2))
    X0_dummy[:,0] = xx
    X0_dummy[:,1] = sex
    
for sid, site in enumerate(cols_site):
    print('configuring dummy data for site',sid, site)
    site_ids = np.zeros((len(xx), len(cols_site)))
    site_ids[:,sid] = 1
    X_dummy = create_design_matrix(X0_dummy, xmin=xmin, xmax=xmax, site_cols=site_ids)
    np.savetxt(os.path.join(data_dir,'cov_bspline_dummy_' + site + '.txt'), X_dummy)

print('configuring dummy data for mean')
site_ids = np.zeros((len(xx), len(cols_site)))
X_dummy = create_design_matrix(X0_dummy, xmin=xmin, xmax=xmax, site_cols=site_ids)
np.savetxt(os.path.join(data_dir,'cov_bspline_dummy_mean.txt'), X_dummy)


blr_metrics = pd.DataFrame(columns = ['eid', 'NLL', 'EV', 'MSLL', 'BIC','Skew','Kurtosis'])
nummer = 0
for idp in idp_ids: 
    nummer = nummer + 1
    print(nummer)
    print('Running IDP:', idp)
    idp_dir = os.path.join(data_dir, idp)
    
    # set output dir 
    out_name = 'blr_' + cov_type
    if warp is not None:
        out_name += '_' + warp
    os.makedirs(os.path.join(idp_dir,out_name), exist_ok=True)
    os.chdir(idp_dir)
    
    # configure and save the responses
    y_tr = df_tr[idp].to_numpy() 
    y_te = df_te[idp].to_numpy()
    
    # remove gross outliers
    yz_tr = (y_tr - np.mean(y_tr)) / np.std(y_tr)
    yz_te = (y_te - np.mean(y_te)) / np.std(y_te)
    nz_tr = np.abs(yz_tr) < outlier_thresh
    nz_te = np.abs(yz_te) < outlier_thresh
    y_tr = y_tr[nz_tr]
    y_te = y_te[nz_te]
    
    resp_file_tr = os.path.join(idp_dir, 'resp_tr.txt')
    resp_file_te = os.path.join(idp_dir, 'resp_te.txt') 
    np.savetxt(resp_file_tr, y_tr)
    np.savetxt(resp_file_te, y_te)
    
    y_tr = y_tr[:, np.newaxis]  
    y_te = y_te[:, np.newaxis]
    
    resp_tr_skew = calibration_descriptives(np.loadtxt(resp_file_tr))[0]
    
    # configure and save the covariates
    X_tr = create_design_matrix(df_tr[cols_cov].loc[nz_tr], site_cols = df_tr[cols_site].loc[nz_tr],
                                basis = 'bspline', xmin = xmin, xmax = xmax)
    X_te = create_design_matrix(df_te[cols_cov].loc[nz_te], site_cols = df_te[cols_site].loc[nz_te],
                                basis = 'bspline', xmin = xmin, xmax = xmax)
    
    np.savetxt(os.path.join(idp_dir, 'cov_bspline_tr.txt'), X_tr)
    np.savetxt(os.path.join(idp_dir, 'cov_bspline_te.txt'), X_te)
    
    # configure the covariates to use
    cov_file_tr = os.path.join(idp_dir, 'cov_') + cov_type + '_tr.txt'
    cov_file_te = os.path.join(idp_dir, 'cov_') + cov_type + '_te.txt'

    w_dir = os.path.join(idp_dir, out_name)
    if not force_refit and os.path.exists(os.path.join(w_dir, 'Models', 'NM_0_0_estimate.pkl')):
        print('Using pre-existing model')
    else:
        w_dir = idp_dir
        if warp == None:
            estimate(cov_file_tr, resp_file_tr, testresp=resp_file_te, 
                     testcov=cov_file_te, alg='blr', configparam=1,
                     optimizer = 'l-bfgs-b', savemodel=True, standardize = False, 
                     hetero_noise = True)
        else: 
             estimate(cov_file_tr, resp_file_tr, testresp=resp_file_te, 
                      testcov=cov_file_te, alg='blr', configparam=1,
                      optimizer = 'l-bfgs-b', savemodel=True, standardize = False, 
                      warp=warp, warp_reparam=True) # if verbose true see inbetween estimates 
    
    # set up the dummy covariates
    cov_file_dummy = os.path.join(data_dir, 'cov_' + cov_type + '_dummy')
    cov_file_dummy = cov_file_dummy + '_mean.txt'
    
    # make predictions
    yhat, s2 = predict(cov_file_dummy, alg='blr', respfile=None, 
                       model_path=os.path.join(w_dir,'Models'))
    
    with open(os.path.join(w_dir,'Models', 'NM_0_0_estimate.pkl'), 'rb') as handle:
        nm = pickle.load(handle) 
    
    # load test data
    yhat_te = np.loadtxt(os.path.join(w_dir, 'yhat_estimate.txt'))
    s2_te = np.loadtxt(os.path.join(w_dir, 'ys2_estimate.txt'))
    yhat_te = yhat_te[:, np.newaxis]
    s2_te = s2_te[:, np.newaxis]
    X_te = np.loadtxt(cov_file_te)
    
    if warp is None:
        # compute evaluation metrics
        metrics = evaluate(y_te, yhat_te)  
        
        # compute MSLL manually as a sanity check
        y_tr_mean = np.array( [[np.mean(y_tr)]] )
        y_tr_var = np.array( [[np.var(y_tr)]] )
        MSLL = compute_MSLL(y_te, yhat_te, s2_te, y_tr_mean, y_tr_var)         
     
    else:
        warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params()+1] 
        W = nm.blr.warp
        
        # warp and plot dummy predictions
        med, pr_int = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param)
        
        # warp predictions
        med_te = W.warp_predictions(np.squeeze(yhat_te), np.squeeze(s2_te), warp_param)[0]
        med_te = med_te[:, np.newaxis]
       
        # evaluation metrics
        metrics = evaluate(y_te, med_te)
        
        # compute MSLL manually
        y_te_w = W.f(y_te, warp_param)
        y_tr_w = W.f(y_tr, warp_param)
        y_tr_mean = np.array( [[np.mean(y_tr_w)]] )
        y_tr_var = np.array( [[np.var(y_tr_w)]] )
        MSLL = compute_MSLL(y_te_w, yhat_te, s2_te, y_tr_mean, y_tr_var)     
  
    y_te_rescaled_all = np.zeros_like(y_te)
    for sid, site in enumerate(cols_site):
                
        # plot the true test data points
        if len(cols_cov) == 1:
            # sex-specific model
            idx = np.where(X_te[:,sid+len(cols_cov)+1] !=0)
        else:
            idx = np.where(np.bitwise_and(X_te[:,2] == sex, X_te[:,sid+len(cols_cov)+1] !=0))
        if len(idx[0]) == 0:
            print('No data for site', sid, site, 'skipping...')
            continue
        else:
            idx_dummy = np.bitwise_and(X_dummy[:,1] > X_te[idx,1].min(), X_dummy[:,1] < X_te[idx,1].max())
        
        # adjust the intercept
        if warp is None:
            y_te_rescaled = y_te[idx] - np.median(y_te[idx]) + np.median(yhat[idx_dummy])
        else:            
            y_te_rescaled = y_te[idx] - np.median(y_te[idx]) + np.median(med[idx_dummy])
        #y_te_rescaled = y_te[idx]
        if show_plot:
            plt.scatter(X_te[idx,1], y_te_rescaled, s=4, color=clr, alpha = 0.05)   
        y_te_rescaled_all[idx] = y_te_rescaled

    if warp is None:
        if show_plot:
            plt.plot(xx, yhat, color = clr)
            plt.fill_between(xx, np.squeeze(yhat-0.67*np.sqrt(s2)), 
                             np.squeeze(yhat+0.67*np.sqrt(s2)), 
                             color=clr, alpha = 0.1)
            plt.fill_between(xx, np.squeeze(yhat-1.64*np.sqrt(s2)), 
                             np.squeeze(yhat+1.64*np.sqrt(s2)), 
                             color=clr, alpha = 0.1)
            plt.fill_between(xx, np.squeeze(yhat-2.33*np.sqrt(s2)), 
                             np.squeeze(yhat+2.32*np.sqrt(s2)), 
                             color=clr, alpha = 0.1)
            plt.plot(xx, np.squeeze(yhat-0.67*np.sqrt(s2)),color=clr, linewidth=0.5)
            plt.plot(xx, np.squeeze(yhat+0.67*np.sqrt(s2)),color=clr, linewidth=0.5)
            plt.plot(xx, np.squeeze(yhat-1.64*np.sqrt(s2)),color=clr, linewidth=0.5)
            plt.plot(xx, np.squeeze(yhat+1.64*np.sqrt(s2)),color=clr, linewidth=0.5)
            plt.plot(xx, np.squeeze(yhat-2.33*np.sqrt(s2)),color=clr, linewidth=0.5)
            plt.plot(xx, np.squeeze(yhat+2.32*np.sqrt(s2)),color=clr, linewidth=0.5)
    else:
        warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params()+1] 
        W = nm.blr.warp
        
        # warp and plot dummy predictions
        med, pr_int = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param)
        
        beta, junk1, junk2 = nm.blr._parse_hyps(nm.blr.hyp, X_dummy)
        s2n = 1/beta
        s2s = s2-s2n
        
        # plot the centiles
        if show_plot: 
            plt.plot(xx, med, clr)
            # fill the gaps in between the centiles
            junk, pr_int25 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.25,0.75])
            junk, pr_int95 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.05,0.95])
            junk, pr_int99 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.01,0.99])
            
            plt.fill_between(xx, pr_int25[:,0], pr_int25[:,1], alpha = 0.1,color=clr)
            plt.fill_between(xx, pr_int95[:,0], pr_int95[:,1], alpha = 0.1,color=clr)
            plt.fill_between(xx, pr_int99[:,0], pr_int99[:,1], alpha = 0.1,color=clr)
            
            # make the width of each line proportional to the epistemic uncertainty
            junk, pr_int25l = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2-0.5*s2s), warp_param, percentiles=[0.25,0.75])
            junk, pr_int95l = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2-0.5*s2s), warp_param, percentiles=[0.05,0.95])
            junk, pr_int99l = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2-0.5*s2s), warp_param, percentiles=[0.01,0.99])
            junk, pr_int25u = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2+0.5*s2s), warp_param, percentiles=[0.25,0.75])
            junk, pr_int95u = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2+0.5*s2s), warp_param, percentiles=[0.05,0.95])
            junk, pr_int99u = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2+0.5*s2s), warp_param, percentiles=[0.01,0.99])    
            plt.fill_between(xx, pr_int25l[:,0], pr_int25u[:,0], alpha = 0.3,color=clr)
            plt.fill_between(xx, pr_int95l[:,0], pr_int95u[:,0], alpha = 0.3,color=clr)
            plt.fill_between(xx, pr_int99l[:,0], pr_int99u[:,0], alpha = 0.3,color=clr)
            plt.fill_between(xx, pr_int25l[:,1], pr_int25u[:,1], alpha = 0.3,color=clr)
            plt.fill_between(xx, pr_int95l[:,1], pr_int95u[:,1], alpha = 0.3,color=clr)
            plt.fill_between(xx, pr_int99l[:,1], pr_int99u[:,1], alpha = 0.3,color=clr)

            # plot ceitle lines
            plt.plot(xx, pr_int25[:,0],color=clr, linewidth=0.5)
            plt.plot(xx, pr_int25[:,1],color=clr, linewidth=0.5)
            plt.plot(xx, pr_int95[:,0],color=clr, linewidth=0.5)
            plt.plot(xx, pr_int95[:,1],color=clr, linewidth=0.5)
            plt.plot(xx, pr_int99[:,0],color=clr, linewidth=0.5)
            plt.plot(xx, pr_int99[:,1],color=clr, linewidth=0.5)
            

    if show_plot:
        plt.xlabel('Age')
        plt.ylabel(idp) 
        plt.title(idp)
        plt.xlim((0,90))
        #plt.ylim((-1000,120000))
        plt.savefig(os.path.join(idp_dir, out_name, 'centiles_' + str(sex)),  bbox_inches='tight')
        plt.show()
     
    BIC = len(nm.blr.hyp) * np.log(y_tr.shape[0]) + 2 * nm.neg_log_lik
    
    # print -log(likelihood)
    print('NLL =', nm.neg_log_lik)
    print('BIC =', BIC)
    print('EV = ', metrics['EXPV'])
    print('MSLL = ', MSLL) 
    
    Z = np.loadtxt(os.path.join(w_dir, 'Z_estimate.txt'))
    [skew, sdskew, kurtosis, sdkurtosis, semean, sesd] = calibration_descriptives(Z)
    
    blr_metrics.loc[len(blr_metrics)] = [idp, nm.neg_log_lik, 
                                         metrics['EXPV'][0], MSLL[0], BIC,
                                         skew, kurtosis]
  
    # save blr stuff
    save_output(idp_dir, os.path.join(idp_dir, out_name))
    
    # if show_plot:
    #     plt.figure()
    #     plt.hist(Z, bins = 100, label = 'skew = ' + str(round(skew,3)) + ' kurtosis = ' + str(round(kurtosis,3)))
    #     plt.title('Z_warp ' + idp)
    #     plt.legend()
    #     plt.savefig(os.path.join(idp_dir, out_name, 'Z_hist'),  bbox_inches='tight')
    #     plt.show()
    
    #     plt.figure()
    #     sm.qqplot(Z, line = '45')
    #     plt.savefig(os.path.join(idp_dir, out_name, 'Z_qq'),  bbox_inches='tight')
    #     plt.show()

#blr_metrics.to_pickle(os.path.join(data_dir,'metrics_' + out_name + '.pkl'))

print(nm.blr.hyp)

