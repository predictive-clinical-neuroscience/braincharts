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

#sys.path.append('/home/preclineu/chafra/Desktop/PCNtoolkit/') # Updated
#sys.path.append('/home/preclineu/andmar/sfw/PCNtoolkit/pcntoolkit/')
from pcntoolkit.normative import estimate, predict, evaluate
from pcntoolkit.util.utils import create_bspline_basis, compute_MSLL

data_dir = '/project_cephfs/3022017.02/projects/big_data/data/ukb_processed_2'
save_im_path = '/project_cephfs/3022017.02/projects/big_data/results/'
#data_dir = '/project_freenas/3022017.02/projects/big_data/data/ukb_processed_top10_dti'
idp_ids = []
with open(os.path.join(data_dir, 'idp_ids.txt'), 'r') as f:
    for line in f:
        idp_ids.append(line.strip())

# Check https://open.win.ox.ac.uk/ukbiobank/big40/BIG40-IDPs_v3/IDPs.html for names
# limit to a couple of interesting cases
#idp_ids = ['Median_Thickness']
#idp_ids = ['25397-2.0'] # OD iFA in m
idp_ids= ['25004-2.0']  # good
#idp_ids = ['25572-2.0'] # good high EV but 'bad' kurtosis
#idp_ids = ['25397-2.0'] #xhard very intresting double curve
#idp_ids = ['25745-2.0'] #x numerical
#idp_ids = ['25884-2.0'] # good
#idp_ids = ['25745-2.0'] #xhard worst fit both b-spline and warp. V hard
#idp_ids = ['25004-2.0'] # good
#idp_ids = ['25781-2.0']  # t2 wmh, good
#idp_ids = ['25572-2.0']  # dmri with outliers *
# idp_ids = ['25162-2.0'] 

#idp_ids = ['25884-2.0'] # IDP_T1_FAST_ROIs, worst kurtosis bspline#
#idp_ids = ['25764-2.0'] # IDP tfMRI 90th percentile zstat faces, worst kurtosis warp
#idp_ids = ['25762-2.0'] # IDP_tfMRI_90th-percentile_zstat_shapes, second worst
# idp_ids = ['25024-2.0'] # IDP_T1_FIRST_right_accumbens_volume, third worst, few outliers present, if removed plot looks okay
# idp_ids = ['25766-2.0'] # IDP_tfMRI_90th-percentile_zstat_faces-shapes, doesn't fit
# idp_ids = ['25046-2.0'] # IDP_tfMRI_median_zstat_faces, very bad skew in warped space
# idp_ids = ['25163-2.0'] # IDP_dMRI_TBSS_MO_Inferior_cerebellar_peduncle_L, not good for both b-spline and warp fit
# idp_ids = ['25763-2.0'] # IDP_tfMRI_90th-percentile_BOLD_faces, strange
# idp_ids = ['25023-2.0'] # IDP_T1_FIRST_left_accumbens_volume, one outlier, if removed plot looks okay
# idp_ids = ['25154-2.0'] # IDP_dMRI_TBSS_MO_Genu_of_corpus_callosum, not good for both spline and warp
# idp_ids = ['25022-2.0'] 

#idp_ids = ['25572-2.0','25397-2.0','25745-2.0','25884-2.0','25745-2.0',
#           '25004-2.0','25781-2.0','25572-2.0','25162-2.0'] 

# which type of model to run?
cov_type = 'bspline'  # 'int', 'bspline' or None
warp = 'WarpSinArcsinh'   # 'WarpBoxCox', 'WarpSinArcsinh'  or None
sex = 0 # 1= female 0 = male
if sex == 0: 
    clr = 'blue';
    sex_name = 'male'
else:
    clr = 'red'
    sex_name = 'female'

# cubic B-spline basis (used for regression)
xmin = 40 # boundaries for ages of UKB participants +/- 5
xmax = 85
B = create_bspline_basis(xmin, xmax)

#################################### FUNCTIONS ################################
def calibration_descriptives(x):
  n = np.shape(x)[0]
  m1 = np.mean(x)
  m2 = sum((x-m1)**2)
  m3 = sum((x-m1)**3)
  m4 = sum((x-m1)**4)
  s1 = np.std(x)
  skew = n*m3/(n-1)/(n-2)/s1**3
  sdskew = np.sqrt( 6*n*(n-1) / ((n-2)*(n+1)*(n+3)) )
  kurtosis = (n*(n+1)*m4 - 3*m2**2*(n-1)) / ((n-1)*(n-2)*(n-3)*s1**4)
  sdkurtosis = np.sqrt( 4*(n**2-1) * sdskew**2 / ((n-3)*(n+5)) )
  semean = np.sqrt(np.var(x)/n)
  sesd = s1/np.sqrt(2*(n-1))
  cd = [skew, sdskew, kurtosis, sdkurtosis, semean, sesd]
  return cd

def save_output(src_dir, dst_dir, savemodel=True):
  
    # move everything else to the destination dir
    files = []
    files.extend(glob.glob(os.path.join(src_dir,'Z*')))
    files.extend(glob.glob(os.path.join(src_dir,'yhat*')))
    files.extend(glob.glob(os.path.join(src_dir,'ys2*')))
    files.extend(glob.glob(os.path.join(src_dir,'Rho*')))
    files.extend(glob.glob(os.path.join(src_dir,'pRho*')))
    files.extend(glob.glob(os.path.join(src_dir,'RMSE*')))
    files.extend(glob.glob(os.path.join(src_dir,'SMSE*')))
    files.extend(glob.glob(os.path.join(src_dir,'MSLL*')))
    files.extend(glob.glob(os.path.join(src_dir,'EXPV*')))
    
    if savemodel:
        model_files = glob.glob(os.path.join(src_dir,'Models/*'))
        dst_model_dir = os.path.join(dst_dir, 'Models')
        os.makedirs(dst_model_dir, exist_ok=True)
        for f in model_files:
            fdir, fnam = os.path.split(f)
            shutil.move(f, os.path.join(dst_model_dir,fnam))
        os.rmdir(os.path.join(src_dir,'Models'))
    else:
        # remove the model directory to save space
        shutil.rmtree(os.path.join(src_dir,'Models'))
    
    for f in files:
        fdir, fnam = os.path.split(f)
        shutil.move(f, os.path.join(dst_dir,fnam))
    return

def test_func(x, epsilon, b):
        return np.sinh(b * np.arcsinh(x) + epsilon * b)

################################### RUN #######################################
# create dummy data for visualisation
xx = np.arange(xmin,xmax,0.5)
X_dummy = np.zeros((len(xx), 3))
X_dummy[:,0] = sex  # male = 1
X_dummy[:,2] = xx
np.savetxt(os.path.join(data_dir,'cov_male_dummy.txt'), X_dummy)
# quadratic expansion
X2_dummy = np.concatenate((X_dummy, np.atleast_2d(X_dummy[:,2]**2).T), axis=1)
x2_dummy_file = os.path.join(data_dir,'cov_male2_dummy.txt')
np.savetxt(x2_dummy_file, xx)
# without intercept
Xz_dummy = np.concatenate((X_dummy, np.array([B(i) for i in X_dummy[:,2]])), axis=1)
np.savetxt(os.path.join(data_dir,'cov_bspline_z_male_dummy.txt'), Xz_dummy)
# with intercept
X_dummy = np.concatenate((X_dummy, np.ones((len(xx), 1))), axis=1)
np.savetxt(os.path.join(data_dir,'cov_int_male_dummy.txt'), X_dummy)
X_dummy = np.concatenate((X_dummy, np.array([B(i) for i in X_dummy[:,2]])), axis=1)
np.savetxt(os.path.join(data_dir,'cov_bspline_male_dummy.txt'), X_dummy)
if cov_type is None:
    cov_file_dummy = os.path.join(data_dir, 'cov_male_dummy.txt')
else:
    cov_file_dummy = os.path.join(data_dir, 'cov_' + cov_type + '_male_dummy.txt')

blr_metrics = pd.DataFrame(columns = ['eid', 'NLL', 'EV', 'MSLL', 'BIC'])
nummer = 0
for idp in idp_ids: 
    nummer = nummer + 1
    print(nummer)
    print('Running IDP:', idp)
    idp_dir = os.path.join(data_dir, idp)
    os.chdir(idp_dir)
    
    # set output dir 
    out_name = 'blr'
    if cov_type is not None:
        out_name += '_' + cov_type
    if warp is not None:
        out_name += '_' + warp
    os.makedirs(os.path.join(idp_dir,out_name), exist_ok=True)
    
    # load data matrices
    X_tr = np.loadtxt(os.path.join(idp_dir, 'cov_tr.txt'))
    X_te = np.loadtxt(os.path.join(idp_dir, 'cov_te.txt'))

    # add intercept column 
    #X_tr = np.concatenate((X_tr, np.ones((X_tr.shape[0],1))), axis=1)
    #X_te = np.concatenate((X_te, np.ones((X_te.shape[0],1))), axis=1)
    #np.savetxt(os.path.join(idp_dir, 'cov_int_tr.txt'), X_tr)
    #np.savetxt(os.path.join(idp_dir, 'cov_int_te.txt'), X_te)
    
    # create Bspline basis set 
    #Phi = np.array([B(i) for i in X_tr[:,2]])
    #Phis = np.array([B(i) for i in X_te[:,2]])
    #X_tr = np.concatenate((X_tr, Phi), axis=1)
    #X_te = np.concatenate((X_te, Phis), axis=1)
    #np.savetxt(os.path.join(idp_dir, 'cov_bspline_tr.txt'), X_tr)
    #np.savetxt(os.path.join(idp_dir, 'cov_bspline_te.txt'), X_te)
    
    # create quadratic expansion of age
    X2_tr = np.concatenate((X_tr, np.atleast_2d(X_tr[:,2]**2).T), axis=1)
    X2_te = np.concatenate((X_te, np.atleast_2d(X_te[:,2]**2).T), axis=1)
    vcov_file_tr = os.path.join(idp_dir, 'cov_x2_tr.txt')
    vcov_file_te = os.path.join(idp_dir, 'cov_x2_te.txt')
    np.savetxt(vcov_file_tr, X_tr[:,2])
    np.savetxt(vcov_file_te, X_te[:,2])
    
    # configure the covariates to use
    if cov_type is None:
        cov_file_tr = os.path.join(idp_dir, 'cov_tr.txt')
        cov_file_te = os.path.join(idp_dir, 'cov_te.txt')
    else:
        cov_file_tr = os.path.join(idp_dir, 'cov_') + cov_type + '_tr.txt'
        cov_file_te = os.path.join(idp_dir, 'cov_') + cov_type + '_te.txt'
    resp_file_tr = os.path.join(idp_dir, 'resp_tr.txt')
    resp_file_te = os.path.join(idp_dir, 'resp_te.txt') 
    
    
    # df_tr = pd.read_csv(resp_file_tr)*100000
    # df_te = pd.read_csv(resp_file_te)*100000

    # df_tr.to_csv('/project_freenas/3022017.02/projects/big_data/data/ukb_processed_2/25291-2.0/resp_tr_multiplied_10000.txt', index= False)
    # df_te.to_csv('/project_freenas/3022017.02/projects/big_data/data/ukb_processed_2/25291-2.0/resp_te_multiplied_10000.txt', index = False)

    # resp_file_tr = os.path.join(idp_dir, 'resp_tr_multiplied_10000.txt')
    # resp_file_te = os.path.join(idp_dir, 'resp_te_multiplied_10000.txt')
    
    resp_tr_skew = calibration_descriptives(np.loadtxt(resp_file_tr))[0]
    
    # run a basic model
    # [mod] specify starting hyperparamters using OLS
    y_tr = np.loadtxt(resp_file_tr)
    y_tr = y_tr[:, np.newaxis]
    Phi_tr = np.loadtxt(cov_file_tr)
    #w_ols = np.linalg.pinv(Phi_tr).dot(y_tr)
    #beta0 = 1/np.var(y_tr- Phi_tr.dot(w_ols))
    #alpha0 = 1/np.var(w_ols)
    #params, params_covariance = optimize.curve_fit(test_func, Phi_tr[:,2].flatten(), y_tr.flatten(),
    #                                           p0=[2, 2])
    
    hyp0 = np.zeros(4)
    #hyp0[1] = params[0]
    #hyp0[2] = params[1]
    #hyp0[0] = np.log(beta0)
    #hyp0[-1] = np.log(alpha0)
    #hyp0[1] = 7.64823948 
    #hyp0[2] = -3.00266243
    #hyp0[-1] = 10
    #hyp0[0] = 10
    
    
    if warp == None:
        estimate(cov_file_tr, resp_file_tr, testresp=resp_file_te, 
              testcov=cov_file_te, alg='blr', configparam=1,
              optimizer = 'powell', savemodel=True, standardize = False, verbose=True,
              varcovfile=vcov_file_tr, l =10)
    else: 
        #hyp0 = np.random.randint(-5,5, 4)
        #hyp0 = [5, -5, 5, -5]
        estimate(cov_file_tr, resp_file_tr, testresp=resp_file_te, 
              testcov=cov_file_te, alg='blr', configparam=1,verbose=True,
              optimizer = 'l-bfgs-b', savemodel=True, standardize = False, 
              warp=warp, warp_reparam=True,
              varcovfile = vcov_file_tr, l =10)
              # hetero_noise=3)
        metrics_new = {'MSLL': 0}


    #nm_new.save('/project_freenas/3022017.02/projects/big_data/data/ukb_processed_2/'+ idp_ids[0] +'/blr_bspline_WarpSinArcsinh/Models/NM_0_0_estimate.pkl' )
    
    # create dummy predictions for visualistion
    # [mod] need to specify alg
    yhat, s2 = predict(cov_file_dummy, alg='blr', respfile=None, testvarcovfile=x2_dummy_file)#hetero_noise=3)

    # [mod] need to add _estimate suffix
    with open(os.path.join(idp_dir, 'Models/','NM_0_0_estimate.pkl'), 'rb') as handle:
       nm = pickle.load(handle)
        
    # load and plot the true test data points
    X_te = np.loadtxt(cov_file_te)
    y_te = np.loadtxt(resp_file_te)
    y_te = y_te[:, np.newaxis]
    idx = np.where(X_te[:,0] == sex)
    plt.figure()
    plt.scatter(X_te[idx,2], y_te[idx], color=clr, alpha = 0.2)
    
    # load training data (needed for MSLL)
    if warp is None:
        plt.plot(xx, yhat, color = clr)
        #plt.fill_between(xx, np.squeeze(yhat-1.96*np.sqrt(s2)), 
        #                 np.squeeze(yhat+1.96*np.sqrt(s2)), 
        #                 color='red', alpha = 0.2)
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
        
        # load test data
        yhat_te = np.loadtxt(os.path.join(idp_dir, 'yhat_estimate.txt'))
        s2_te = np.loadtxt(os.path.join(idp_dir, 'ys2_estimate.txt'))
        yhat_te = yhat_te[:, np.newaxis]
        s2_te = s2_te[:, np.newaxis]
        
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
        junk, pr_int25 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.25,0.75])
        junk, pr_int95 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.05,0.95])
        junk, pr_int99 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.01,0.99])
        plt.plot(xx, med, clr)
        #plt.fill_between(xx, pr_int[:,0], pr_int[:,1], alpha = 0.2,color=clr)
        plt.fill_between(xx, pr_int25[:,0], pr_int25[:,1], alpha = 0.1,color=clr)
        plt.fill_between(xx, pr_int95[:,0], pr_int95[:,1], alpha = 0.1,color=clr)
        plt.fill_between(xx, pr_int99[:,0], pr_int99[:,1], alpha = 0.1,color=clr)
        plt.plot(xx, pr_int25[:,0],color=clr, linewidth=0.5)
        plt.plot(xx, pr_int25[:,1],color=clr, linewidth=0.5)
        plt.plot(xx, pr_int95[:,0],color=clr, linewidth=0.5)
        plt.plot(xx, pr_int95[:,1],color=clr, linewidth=0.5)
        plt.plot(xx, pr_int99[:,0],color=clr, linewidth=0.5)
        plt.plot(xx, pr_int99[:,1],color=clr, linewidth=0.5)
        
        # load test data
        yhat_te = np.loadtxt(os.path.join(idp_dir, 'yhat_estimate.txt'))
        s2_te = np.loadtxt(os.path.join(idp_dir, 'ys2_estimate.txt'))
        yhat_te = yhat_te[:, np.newaxis]
        s2_te = s2_te[:, np.newaxis]
        
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
        
    plt.xlim((45,80))
    if idp_ids == ['25046-2.0']:
        plt.ylim(round(y_te.min()*2),round(y_te.max()*2))
        plt.xlabel('Age')
        plt.ylabel('WMH volume ($mm^3$)') 
    else:
        plt.xlabel('Age')
        plt.ylabel(idp) 
    plt.savefig(os.path.join(idp_dir, out_name, 'normative_model_plot'),  bbox_inches='tight')
# if warp == 'WarpSinArcsinh':
#     plt.savefig(save_im_path +'idp_'+ idp_ids[0]+ '_warp_'+sex_name+'.png',  bbox_inches='tight')
# else:
#     plt.savefig(save_im_path +'idp_'+ idp_ids[0]+ '_bspline_'+sex_name+'.png',  bbox_inches='tight')
    plt.show()
#plt.clf()

    BIC = len(nm.blr.hyp) * np.log(y_tr.shape[0]) + 2 * nm.neg_log_lik
    
    # print -log(likelihood)
    print('NLL =', nm.neg_log_lik)
    print('BIC =', BIC)
    print('EV = ', metrics['EXPV'])
    print('MSLL = ', MSLL) 
    
    blr_metrics.loc[len(blr_metrics)] = [idp, nm.neg_log_lik, 
                                         metrics['EXPV'][0], MSLL[0], BIC]
  
    # save blr stuff
    # 
    save_output(idp_dir, os.path.join(idp_dir, out_name))
    Z = np.loadtxt(os.path.join(idp_dir, out_name, 'Z_estimate.txt'))
    # remove biggest outlier
    # Z = Z[Z!=Z.min()]
    [skew, sdskew, kurtosis, sdkurtosis, semean, sesd] = calibration_descriptives(Z)
    plt.figure()
    plt.hist(Z, bins = 100, label = 'skew = ' + str(round(skew,3)) + ' kurtosis = ' + str(round(kurtosis,3)))
    plt.title('Z_warp ' + idp)
    plt.legend()
    plt.savefig(os.path.join(idp_dir, out_name, 'Z_hist'),  bbox_inches='tight')
    plt.show()
    
    plt.figure()
    sm.qqplot(Z, line = '45')
    plt.savefig(os.path.join(idp_dir, out_name, 'Z_qq'),  bbox_inches='tight')
    plt.show()
#blr_metrics.to_pickle(os.path.join(data_dir,'metrics_' + out_name + '.pkl'))

#################################### CHECK KURTOSIS AND SKEW ##################
if warp is None:
    out_name_bspline = 'blr_bspline'
    Z = np.loadtxt(os.path.join(idp_dir, out_name, 'Z_estimate.txt'))
    [skew, sdskew, kurtosis, sdkurtosis, semean, sesd] = calibration_descriptives(Z)
    #plt.figure()
    #plt.hist(Z, bins = 100, label = 'skew = ' + str(round(skew,3)) + ' kurtosis = ' + str(round(kurtosis,3)))
    #plt.title('Z')
    #plt.legend()
    
    te = np.loadtxt(os.path.join(idp_dir, 'resp_te.txt'))
    #plt.figure()
    #plt.hist(te, bins = 100)
    #plt.title('te')
    
    #sm.qqplot(Z, line = '45')
    
    yhat = np.loadtxt(os.path.join(idp_dir, out_name, 'yhat_estimate.txt'))
    #plt.figure()
    #plt.hist(yhat, bins = 100)
    #plt.title('Yhat')
    print(nm.blr.hyp)

else: 
    
    Z = np.loadtxt(os.path.join(idp_dir, out_name, 'Z_estimate.txt'))
    # remove biggest outlier
    # Z = Z[Z!=Z.min()]
    [skew, sdskew, kurtosis, sdkurtosis, semean, sesd] = calibration_descriptives(Z)
    #plt.figure()
    #plt.hist(Z, bins = 100, label = 'skew = ' + str(round(skew,3)) + ' kurtosis = ' + str(round(kurtosis,3)))
    #plt.title('Z_warp')
    #plt.legend()
    
    #sm.qqplot(Z, line = '45')
    
    te = np.loadtxt(os.path.join(idp_dir, 'resp_te.txt'))
    Ywarp = nm.blr.warp.f(te, warp_param)
    #plt.figure()
    #plt.hist(Ywarp, bins = 100)
    #plt.title('Ywarp')
    
    yhat = np.loadtxt(os.path.join(idp_dir, out_name, 'yhat_estimate.txt'))
    #plt.figure()
    #plt.hist(yhat, bins = 100)
    #plt.title('Yhat')
    
    ys2 = np.loadtxt(os.path.join(idp_dir, out_name, 'ys2_estimate.txt'))
    #plt.figure()
    #plt.hist(ys2, bins = 100)
    #plt.title('ys2')
    
    #Z_calculated = (Ywarp-yhat)/np.sqrt(ys2)
    #plt.figure()
    #plt.hist(Z_calculated, bins = 100)
    #plt.title('Z_te')
    
    #tr = np.loadtxt(resp_file_tr)
    #Ywarp_tr = nm.blr.warp.f(tr, warp_param)
    #plt.figure()
    #plt.hist(Ywarp_tr, bins = 100)
    #plt.title('Ywarp_tr')
    #yhat_tr = [yhat[0]]* len(Ywarp_tr)
    #ys2_tr = [ys2[0]] * len(Ywarp_tr)
    
    #Z_train_data = (Ywarp_tr-yhat_tr)/np.sqrt(ys2_tr)
    #plt.figure()
    #plt.hist(Z_train_data, bins = 100)
    #plt.title('Z_tr')
    
    #resp_tr = np.loadtxt(os.path.join(idp_dir, 'resp_tr.txt')) 
    #plt.figure()
    #plt.hist(resp_tr, bins = 100)
    #plt.title('Y_tr')    
    
    print(warp_param)

print(nm.blr.hyp)

