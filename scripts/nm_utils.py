import os
import glob
import numpy as np
import pandas as pd
import shutil
import pickle

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
        if os.path.exists(os.path.join(src_dir,'Models')):
            os.rmdir(os.path.join(src_dir,'Models'))
    else:
        if os.path.exists(os.path.join(src_dir,'Models')):
            # remove the model directory to save space
            shutil.rmtree(os.path.join(src_dir,'Models'))
    
    for f in files:
        fdir, fnam = os.path.split(f)
        shutil.move(f, os.path.join(dst_dir,fnam))
    return

def test_func(x, epsilon, b):
        return np.sinh(b * np.arcsinh(x) + epsilon * b)
    

def remove_bad_subjects(df, qc):#qc_file):
    
    """
    Removes low-quality subjects from multi-site data based on Euler characteristic 
    measure.
    
    * Inputs:
        - df: the data in a pandas' dataframe format.
        - qc_file: the address of pickle file containing the euler charcteristics.
    
    * Outputs:
        - df: the updated data after removing bad subjects.
        - removed_subjects: the list of removed subjects.
    """
    
    n = df.shape[0]
    
    #with open(qc_file, 'rb') as file:
    #    qc = pickle.load(file)['qc_all']
    #
    #qc = qc.reindex(df.index, fill_value=0)
    
    euler_nums = qc['avg_en'].to_numpy(dtype=np.float32)
    #sites = df['site'].to_numpy(dtype=np.int)
    site_ids = pd.Series(df['site'], copy=True)
    for i,s in enumerate(site_ids.unique()):
        site_ids.loc[site_ids == s] = i
    sites = site_ids.to_numpy(dtype=np.int)
    subjects = qc.index
    
    for site in np.unique(sites):
        euler_nums[sites==site] = np.sqrt(-(euler_nums[sites==site])) - np.median(np.sqrt(-(euler_nums[sites==site])))
    
    good_subjects = list(subjects[np.bitwise_or(euler_nums<=5, np.isnan(euler_nums))])
    removed_subjects = list(subjects[euler_nums>5])
    
    good_subjects = list(set(good_subjects))
    
    #df = df.reindex(good_subjects)
    #df = df.dropna()
    dfout = df.loc[good_subjects]
    
    
    print('%f of subjects are removed!'  %((n - df.shape[0]) / n )) 
    
    return dfout, removed_subjects
