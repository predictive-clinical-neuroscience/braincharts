import os
import glob
import numpy as np
import shutil

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