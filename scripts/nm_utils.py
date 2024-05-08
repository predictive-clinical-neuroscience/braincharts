import os
import glob
import re
import numpy as np
import pandas as pd
import shutil
import pickle
import subprocess

import pcntoolkit.dataio.fileio as fileio

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

def predict_on_new_sites(blr, hyp, X, y, Xs=None, 
                         ys=None, 
                         var_groups_test=None):
    """ Function to transfer the model to a new site"""
    # Get predictions from old model on new data X
    ys_ref, s2_ref = blr.predict(hyp, None, None, X)

    # Subtract the predictions from true data to get the residuals
    if blr.warp is None:
        residuals = ys_ref-y
    else:
        # Calculate the residuals in warped space
        y_ref_ws = blr.warp.f(y, hyp[1:blr.warp.get_n_params()+1])
        residuals = ys_ref - y_ref_ws 
  
    residuals_mu = np.mean(residuals)
    residuals_sd = np.std(residuals)

    # Adjust the mean with the mean of the residuals
    #blr.m = blr.m-np.ones((len(blr.m)))*residuals_mu 
    #ys,s2 = blr.predict(hyp, None, None, Xs)
    if ys is None:
        if Xs is None:
            raise(ValueError, 'Either ys or Xs must be specified')
        else:
            ys, s2 = blr.predict(hyp, None, None, Xs)
            ys = ys - residuals_mu 
    else:
        if blr.warp is not None:
            y_ws = blr.warp.f(y, hyp[1:blr.warp.get_n_params()+1])
            ys = y_ws - residuals_mu 
        else:
            ys = ys - residuals_mu    
        
    # Set the deviation to the devations of the residuals
    s2 = np.ones(len(s2))*residuals_sd**2
        
    return ys, s2
        

def test_func(x, epsilon, b):
        return np.sinh(b * np.arcsinh(x) + epsilon * b)
    
def remove_bad_subjects(df, qc):#qc_file):
    
    """
    Removes low-quality subjects from multi-site data based on Euler characteristic 
    measure.
    
    * Inputs:
        - df: the data in a pandas' dataframe format.
        - qc: pandas dataframe containing the euler charcteristics.
    
    * Outputs:
        - df: the updated data after removing bad subjects.
        - removed_subjects: the list of removed subjects.
    """
    
    n = df.shape[0]
    
    euler_nums = qc['avg_en'].to_numpy(dtype=np.float32)
    # convert to numeric site indices
    #sites = df['site'].to_numpy(dtype=np.int)
    site_ids = pd.Series(df['site'], copy=True)
    for i,s in enumerate(site_ids.unique()):
        site_ids.loc[site_ids == s] = i
    sites = site_ids.to_numpy(dtype=np.int32)
    subjects = qc.index
    for site in np.unique(sites):
        euler_nums[sites==site] = np.sqrt(-(euler_nums[sites==site])) - np.nanmedian(np.sqrt(-(euler_nums[sites==site])))
    
    good_subjects = list(subjects[np.bitwise_or(euler_nums<=5, np.isnan(euler_nums))])
    removed_subjects = list(subjects[euler_nums>5])
    
    good_subjects = list(set(good_subjects))
    
    dfout = df.loc[good_subjects]
    
    print(len(removed_subjects), 'subjects are removed!') 
    
    return dfout, removed_subjects

def retrieve_eulernum(freesurfer_dir, subjects=None):
    """ Get the Euler Characteristic from a set of subjects
        :param freesurfer_dir: Freesurfer SUBJECTS_DIR
        :param subjects: a list of subjects to process
    """
    
    if subjects is None:
        subjects = [temp for temp in os.listdir(freesurfer_dir) 
                    if os.path.isdir(os.path.join(freesurfer_dir ,temp))]
        
    df = pd.DataFrame(index=subjects, columns=['lh_en','rh_en','avg_en'])
    missing_subjects = []
    
    for s, sub in enumerate(subjects):
        sub_dir = os.path.join(freesurfer_dir, sub)
        log_file = os.path.join(sub_dir, 'scripts', 'recon-all.log')
        
        if os.path.exists(sub_dir):
            if os.path.exists(log_file):    
                with open(log_file) as f:
                    for line in f:
                        # find the part that refers to the EC
                        if re.search('orig.nofix lheno', line):
                            eno_line = line
                f.close()
                eno_l = eno_line.split()[3][0:-1] # remove the trailing comma
                eno_r = eno_line.split()[6]
                euler = (float(eno_l) + float(eno_r)) / 2
                
                df.at[sub, 'lh_en'] = eno_l
                df.at[sub, 'rh_en'] = eno_r
                df.at[sub, 'avg_en'] = euler
                
                print('%d: Subject %s is successfully processed. EN = %f' 
                      %(s, sub, df.at[sub, 'avg_en']))
            else:
                print('%d: Subject %s is missing log file, running QC ...' %(s, sub))
                try:
                    bashCommand = 'mris_euler_number '+ freesurfer_dir + sub +'/surf/lh.orig.nofix>' + 'temp_l.txt 2>&1'
                    res = subprocess.run(bashCommand, stdout=subprocess.PIPE, shell=True)
                    file = open('temp_l.txt', mode = 'r', encoding = 'utf-8-sig')
                    lines = file.readlines()
                    file.close()
                    words = []
                    for line in lines:
                        line = line.strip()
                        words.append([item.strip() for item in line.split(' ')])
                    eno_l = np.float32(words[0][12])
                    
                    bashCommand = 'mris_euler_number '+ freesurfer_dir + sub +'/surf/rh.orig.nofix>' + 'temp_r.txt 2>&1'
                    res = subprocess.run(bashCommand, stdout=subprocess.PIPE, shell=True)
                    file = open('temp_r.txt', mode = 'r', encoding = 'utf-8-sig')
                    lines = file.readlines()
                    file.close()
                    words = []
                    for line in lines:
                        line = line.strip()
                        words.append([item.strip() for item in line.split(' ')])
                    eno_r = np.float32(words[0][12])
                    
                    df.at[sub, 'lh_en'] = eno_l
                    df.at[sub, 'rh_en'] = eno_r
                    df.at[sub, 'avg_en'] = (eno_r + eno_l) / 2
                
                    print('%d: Subject %s is successfully processed. EN = %f' 
                          %(s, sub, df.at[sub, 'avg_en']))
                    
                except:
                    missing_subjects.append(sub)
                    print('%d: QC is failed for subject %s.' %(s, sub))
                
        else:
            missing_subjects.append(sub)
            print('%d: Subject %s is missing.' %(s, sub))
        df = df.dropna()
             
    return df, missing_subjects

def load_2d(filename):
    """ this simple function loads a data type supported by PCNtoolkit and
        ensures that the output is a 2d numpy array
    """
    
    x = fileio.load(filename)
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    
    return x
