import os
import numpy as np
import pandas as pd
from ukb_utils import get_variables_UKB, lookup_UKB

data_dir = '/home/preclineu/andmar/data/sairut/data'

dem_ids = ['21003-2.0', '31-0.0', '54-2.0']
dem_nam = ['age', 'sex', 'site_id']
    
# derive from category 190, 192 and 197 in UKB
subcort_dic = {'BrainSegVol' : 26514,
               'BrainSegVolNotVentSurf' : 26516,
               'SubCortGrayVol' : 26517,
               'TotalGrayVol' :  26518,
               'SupraTentorialVol' :  26519,
               'SupraTentorialVolNotVent' :  26520,
               'EstimatedTotalIntraCranialVol' : 26521,
               '3rd-Ventricle' : 26523,
               '4th-Ventricle' : 26524,
               '5th-Ventricle' : 26525,
               'Brain-Stem' : 26526,
               'CSF' : 26527 ,'WM-hypointensities' : 26528,
               'non-WM-hypointensities' : 26529,
               'Optic-Chiasm' : 26530,
               'CC_Posterior' : 26531,
               'CC_Mid_Posterior' : 26532,
               'CC_Central' : 26533,
               'CC_Mid_Anterior' : 26534,
               'CC_Anterior' : 26535,
               'BrainSegVol-to-eTIV' : 26536,
               'MaskVol-to-eTIV' : 26537,
               'lhCortexVol' : 26552,
               'lhCerebralWhiteMatterVol' : 26553,
               'Left-Lateral-Ventricle' : 26554,
               'Left-Inf-Lat-Vent' : 26555,
               'Left-Cerebellum-White-Matter' : 26556,
               'Left-Cerebellum-Cortex' : 26557,
               'Left-Thalamus-Proper' : 26558,
               'Left-Caudate' : 26559,
               'Left-Putamen' : 26560,
               'Left-Pallidum' : 26561,
               'Left-Hippocampus' : 26562,
               'Left-Amygdala' : 26563,
               'Left-Accumbens-area' : 26564,
               'Left-VentralDC' : 26565,
               'Left-vessel' : 26566,
               'Left-choroid-plexus' : 26567,
               'lhSurfaceHoles' : 26568,
               'rhCortexVol' : 26583,
               'rhCerebralWhiteMatterVol' : 26584,
               'Right-Lateral-Ventricle' : 26585,
               'Right-Inf-Lat-Vent' : 26586,
               'Right-Cerebellum-White-Matter' : 26587,
               'Right-Cerebellum-Cortex' : 26588,
               'Right-Thalamus-Proper' : 26589,
               'Right-Caudate' : 26590,'Right-Putamen' : 26591,
               'Right-Pallidum' : 26592,
               'Right-Hippocampus' : 26593,
               'Right-Amygdala' : 26594,
               'Right-Accumbens-area' : 26595,
               'Right-VentralDC' : 26596,
               'Right-vessel' : 26597,
               'Right-choroid-plexus' : 26598,
               'rhSurfaceHoles' : 26599}
sc_ids = []
sc_nam = []
for key, value in subcort_dic.items():
    sc_nam.append(key)
    sc_ids.append(value) 
 
# from category 
lh_ids = list(range(27403,27477)) + [26755]
lh_nam = ['lh_G&S_frontomargin_thickness',
          'lh_G&S_occipital_inf_thickness',
          'lh_G&S_paracentral_thickness',
          'lh_G&S_subcentral_thickness',
          'lh_G&S_transv_frontopol_thickness',
          'lh_G&S_cingul-Ant_thickness',
          'lh_G&S_cingul-Mid-Ant_thickness',
          'lh_G&S_cingul-Mid-Post_thickness',
          'lh_G_cingul-Post-dorsal_thickness',
          'lh_G_cingul-Post-ventral_thickness',
          'lh_G_cuneus_thickness',
          'lh_G_front_inf-Opercular_thickness',
          'lh_G_front_inf-Orbital_thickness',
          'lh_G_front_inf-Triangul_thickness',
          'lh_G_front_middle_thickness',
          'lh_G_front_sup_thickness',
          'lh_G_Ins_lg&S_cent_ins_thickness',
          'lh_G_insular_short_thickness',
          'lh_G_occipital_middle_thickness',
          'lh_G_occipital_sup_thickness',
          'lh_G_oc-temp_lat-fusifor_thickness',
          'lh_G_oc-temp_med-Lingual_thickness',
          'lh_G_oc-temp_med-Parahip_thickness',
          'lh_G_orbital_thickness',
          'lh_G_pariet_inf-Angular_thickness',
          'lh_G_pariet_inf-Supramar_thickness',
          'lh_G_parietal_sup_thickness',
          'lh_G_postcentral_thickness',
          'lh_G_precentral_thickness',
          'lh_G_precuneus_thickness',
          'lh_G_rectus_thickness',
          'lh_G_subcallosal_thickness',
          'lh_G_temp_sup-G_T_transv_thickness',
          'lh_G_temp_sup-Lateral_thickness',
          'lh_G_temp_sup-Plan_polar_thickness',
          'lh_G_temp_sup-Plan_tempo_thickness',
          'lh_G_temporal_inf_thickness',
          'lh_G_temporal_middle_thickness',
          'lh_Lat_Fis-ant-Horizont_thickness',
          'lh_Lat_Fis-ant-Vertical_thickness',
          'lh_Lat_Fis-post_thickness',
          'lh_Pole_occipital_thickness',
          'lh_Pole_temporal_thickness',
          'lh_S_calcarine_thickness',
          'lh_S_central_thickness',
          'lh_S_cingul-Marginalis_thickness',
          'lh_S_circular_insula_ant_thickness',
          'lh_S_circular_insula_inf_thickness',
          'lh_S_circular_insula_sup_thickness',
          'lh_S_collat_transv_ant_thickness',
          'lh_S_collat_transv_post_thickness',
          'lh_S_front_inf_thickness',
          'lh_S_front_middle_thickness',
          'lh_S_front_sup_thickness',
          'lh_S_interm_prim-Jensen_thickness',
          'lh_S_intrapariet&P_trans_thickness',
          'lh_S_oc_middle&Lunatus_thickness',
          'lh_S_oc_sup&transversal_thickness',
          'lh_S_occipital_ant_thickness',
          'lh_S_oc-temp_lat_thickness',
          'lh_S_oc-temp_med&Lingual_thickness',
          'lh_S_orbital_lateral_thickness',
          'lh_S_orbital_med-olfact_thickness',
          'lh_S_orbital-H_Shaped_thickness',
          'lh_S_parieto_occipital_thickness',
          'lh_S_pericallosal_thickness',
          'lh_S_postcentral_thickness',
          'lh_S_precentral-inf-part_thickness',
          'lh_S_precentral-sup-part_thickness',
          'lh_S_suborbital_thickness',
          'lh_S_subparietal_thickness',
          'lh_S_temporal_inf_thickness',
          'lh_S_temporal_sup_thickness',
          'lh_S_temporal_transverse_thickness',
          'lh_MeanThickness_thickness']

rh_ids = list(range(27625,27699)) + [26856]
rh_nam = ['rh_G&S_frontomargin_thickness',
          'rh_G&S_occipital_inf_thickness',
          'rh_G&S_paracentral_thickness',
          'rh_G&S_subcentral_thickness', 
          'rh_G&S_transv_frontopol_thickness',
          'rh_G&S_cingul-Ant_thickness', 
          'rh_G&S_cingul-Mid-Ant_thickness', 
          'rh_G&S_cingul-Mid-Post_thickness', 
          'rh_G_cingul-Post-dorsal_thickness', 
          'rh_G_cingul-Post-ventral_thickness',
          'rh_G_cuneus_thickness', 
          'rh_G_front_inf-Opercular_thickness',
          'rh_G_front_inf-Orbital_thickness', 
          'rh_G_front_inf-Triangul_thickness',
          'rh_G_front_middle_thickness', 
          'rh_G_front_sup_thickness', 
          'rh_G_Ins_lg&S_cent_ins_thickness', 
          'rh_G_insular_short_thickness', 
          'rh_G_occipital_middle_thickness',
          'rh_G_occipital_sup_thickness', 
          'rh_G_oc-temp_lat-fusifor_thickness',
          'rh_G_oc-temp_med-Lingual_thickness', 
          'rh_G_oc-temp_med-Parahip_thickness', 
          'rh_G_orbital_thickness', 
          'rh_G_pariet_inf-Angular_thickness',
          'rh_G_pariet_inf-Supramar_thickness', 
          'rh_G_parietal_sup_thickness', 
          'rh_G_postcentral_thickness', 
          'rh_G_precentral_thickness', 
          'rh_G_precuneus_thickness',
          'rh_G_rectus_thickness',
          'rh_G_subcallosal_thickness',
          'rh_G_temp_sup-G_T_transv_thickness', 
          'rh_G_temp_sup-Lateral_thickness', 
          'rh_G_temp_sup-Plan_polar_thickness',
          'rh_G_temp_sup-Plan_tempo_thickness',
          'rh_G_temporal_inf_thickness', 
          'rh_G_temporal_middle_thickness',
          'rh_Lat_Fis-ant-Horizont_thickness',
          'rh_Lat_Fis-ant-Vertical_thickness', 
          'rh_Lat_Fis-post_thickness', 
          'rh_Pole_occipital_thickness', 
          'rh_Pole_temporal_thickness', 
          'rh_S_calcarine_thickness', 
          'rh_S_central_thickness', 
          'rh_S_cingul-Marginalis_thickness', 
          'rh_S_circular_insula_ant_thickness',
          'rh_S_circular_insula_inf_thickness', 
          'rh_S_circular_insula_sup_thickness',
          'rh_S_collat_transv_ant_thickness', 
          'rh_S_collat_transv_post_thickness',
          'rh_S_front_inf_thickness',
          'rh_S_front_middle_thickness', 
          'rh_S_front_sup_thickness', 
          'rh_S_interm_prim-Jensen_thickness', 
          'rh_S_intrapariet&P_trans_thickness',
          'rh_S_oc_middle&Lunatus_thickness', 
          'rh_S_oc_sup&transversal_thickness',
          'rh_S_occipital_ant_thickness',
          'rh_S_oc-temp_lat_thickness', 
          'rh_S_oc-temp_med&Lingual_thickness', 
          'rh_S_orbital_lateral_thickness',
          'rh_S_orbital_med-olfact_thickness',
          'rh_S_orbital-H_Shaped_thickness', 
          'rh_S_parieto_occipital_thickness',
          'rh_S_pericallosal_thickness',
          'rh_S_postcentral_thickness',
          'rh_S_precentral-inf-part_thickness',
          'rh_S_precentral-sup-part_thickness',
          'rh_S_suborbital_thickness',
          'rh_S_subparietal_thickness',
          'rh_S_temporal_inf_thickness',
          'rh_S_temporal_sup_thickness',
          'rh_S_temporal_transverse_thickness',
          'rh_MeanThickness_thickness']

# missing from category 190:
# see https://surfer.nmr.mgh.harvard.edu/fswiki/MorphometryStats
drop_cols = ['Left-WM-hypointensities',
            'Right-WM-hypointensities',
            'Left-non-WM-hypointensities',
            'Right-non-WM-hypointensities',
            'CortexVol',
            'CerebralWhiteMatterVol',
            'SupraTentorialVolNotVentVox',
            'MaskVol',
            'SurfaceHoles',
            'IntraCranialVol']

# missing from category 197:
drop_cols = drop_cols + ['BrainSegVolNotVent', 'eTIV']# (= 'EstimatedTotalIntraCranialVol')

# add instance ids
lh_idps = [str(x)+'-2.0' for x in lh_ids]
rh_idps = [str(x)+'-2.0' for x in rh_ids]
sc_idps = [str(x)+'-2.0' for x in sc_ids]

field_ids = lh_idps + rh_idps + sc_idps
field_names = lh_nam + rh_nam + sc_nam 

# processing  existing data 
df_tr = pd.read_csv(os.path.join(data_dir,'lifespan_tr.csv'), index_col=0) 
df_te = pd.read_csv(os.path.join(data_dir,'lifespan_te.csv'), index_col=0)
cols_site = df_te.columns.to_list()[234:270]

for col in drop_cols:
    df_tr.drop(col,1,inplace=True)
    df_te.drop(col,1,inplace=True)

df_te.drop(df_te[df_te['site_ID'] == 'ukb-11025.0'].index, inplace=True)
df_te.drop(df_te[df_te['site_ID'] == 'ukb-11027.0'].index, inplace=True)
df_tr.drop(df_tr[df_tr['site_ID'] == 'ukb-11025.0'].index, inplace=True)
df_tr.drop(df_tr[df_tr['site_ID'] == 'ukb-11027.0'].index, inplace=True)

# add extra site column
df_tr['site_ukb-11026.0'] = 0
df_te['site_ukb-11026.0'] = 0

# change sex coding
df_te['sex'] = df_te['sex'] - 1
df_tr['sex'] = df_tr['sex'] - 1

# load the UKB data
csv_file_path = '/project_freenas/3022017.02/UKB/phenotypes/ukb43578.csv'
save_path =  '/project/3022000.05/sairut/data/fs_ukb.csv'
idps, sub_idps = get_variables_UKB(csv_file_path, field_ids, field_names, save_path)

csv_file_path = '/project_freenas/3022017.02/UKB/phenotypes/ukb42006.csv'
save_path =  '/project/3022000.05/sairut/data/dem_ukb.csv'
dem, sub_dem = get_variables_UKB(csv_file_path, dem_ids, dem_nam, save_path)

df_ukb = dem.join(idps)
df_ukb.dropna(how='any',inplace=True)
df_ukb.rename(columns={'site_id': 'site_ID'}, inplace = True)

# add existing site columns
for site in cols_site:
    df_ukb[site] = 0

# add UKB site columns
ukb_sites = []
for site in df_ukb['site_ID'].sort_values().unique():
    ukb_str = 'site_ukb-'+str(site)
    df_ukb[ukb_str] = (df_ukb['site_ID'] == site).astype(int)
    ukb_sites.append(ukb_str)

sid = max(df_te['site_ID_bin']) + 1
df_ukb['site_ID_bin'] = 0
for site in ukb_sites:    
    df_ukb['site_ID_bin'].loc[df_ukb[site] == 1]= sid  
    sid = sid + 1
#del df_ukb['site_id']

# add participant id column 
df_ukb['participant_id'] = df_ukb.index
df_ukb['avg_thickness'] = (df_ukb['rh_MeanThickness_thickness'] + \
                           df_ukb['lh_MeanThickness_thickness']) / 2.0

# split, concatenate with original data and save
tr = np.random.uniform(size=df_ukb.shape[0]) > 0.5
te = ~tr

df_te_final = pd.concat((df_te, df_ukb.loc[te]))
df_tr_final = pd.concat((df_tr, df_ukb.loc[tr]))

# rearrange columns 
cols = df_te_final.columns.to_list()
newcols = cols[:-4] + [cols[-1]] + [cols[-4]] + [cols[-3]] + [cols[-2]]
df_te_final = df_te_final[newcols]
df_tr_final = df_tr_final[newcols]

df_te_final.to_csv(os.path.join(data_dir,'lifespan_full_te.csv'))
df_tr_final.to_csv(os.path.join(data_dir,'lifespan_full_tr.csv'))

