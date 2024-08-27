import os
import pandas as pd
import numpy as np
from scipy.io import loadmat, savemat

# file path
gmv_path = './AAL_1_5mm_Output_Region_GMV.csv'
basc_235_gmv_path = './rtemplate_cambridge_basc_multiscale_asym_scale325_Output_Region_GMV.csv'
cc_200_gmv_path = './rADHD200_parcellate_200_Output_Region_GMV.csv'
bn_274_gmv_path = './BN_Atlas_274_with_cerebellum_without_255_1_5mm_Output_Region_GMV.csv'
xlsx_973_path = './data/PANSS-SZ_973_1100.xlsx'
sfc_path = 'data/tz_sfc.mat'  # (1100,1225)
tc_path = 'data/tz_tc_norm.mat'

output_m = 'align_fnc_tc_gmv.mat'

if os.path.exists(output_m):
    os.remove(output_m)

def get_gmv(gmv_path, individual):
    # load GMV
    data_gmv = pd.read_csv(gmv_path)
    first_column = np.array(data_gmv.iloc[:, 0])
    data_gmv_column = np.array(data_gmv.iloc[:, 1:])
    truncated_arr = np.array([s[6:16] for s in first_column])
    truncated_arr_set = set(truncated_arr)
    # computing difference
    # gmv - 1100: in gmv, not in 1100
    # 1100 - gmv: in 1100, not in gmv
    intersection = truncated_arr_set & set(individual)
    intersection_id = np.array(list(intersection))
    # get index from individual
    indices = np.where(np.isin(individual, intersection_id))[0]
    # names = individual[indices]
    return data_gmv_column, truncated_arr, indices

# load ID
excel_data = pd.read_excel(xlsx_973_path)
name_lists = excel_data.iloc[:, 1]
name_lists = np.array(name_lists)
name_lists = set(name_lists)
# print(not_in)

# load FNC and TCs
data_fnc = loadmat(sfc_path)
X_fnc = data_fnc.get('data')
y = data_fnc.get('label').T
data_tc = loadmat(tc_path)
X_tc = data_tc.get('tc_data_170')
individual = data_tc.get('individual')
individual_set = set(individual)

# Match/Save.
data_gmv_column, truncated_arr, indices = get_gmv(gmv_path, individual)
bn_data_gmv_column, bn_truncated_arr, _ = get_gmv(bn_274_gmv_path, individual)
cc_200_data_gmv_column, cc_200_truncated_arr, _ = get_gmv(cc_200_gmv_path, individual)
basc_235_data_gmv_column, basc_235_truncated_arr, _ = get_gmv(basc_235_gmv_path, individual)

# align data by ID
align_fnc_all = list()
align_tc_all = list()
align_gmv_all = list()
align_bn_gmv_all = list()
align_cc_200_gmv_all = list()
align_basc_235_gmv_all = list()
align_y_all = list()

for id in indices:
    fnc_item = X_fnc[id]
    tc_item = X_tc[id]
    y_item = y[id]
    gmv_item = data_gmv_column[np.where(truncated_arr==[individual[id]])[0][0]]
    bn_gmv_item = bn_data_gmv_column[np.where(bn_truncated_arr == [individual[id]])[0][0]]
    cc_200_gmv_item = cc_200_data_gmv_column[np.where(cc_200_truncated_arr == [individual[id]])[0][0]]
    basc_235_gmv_item = basc_235_data_gmv_column[np.where(basc_235_truncated_arr == [individual[id]])[0][0]]

    align_fnc_all.append(fnc_item)
    align_tc_all.append(tc_item)
    align_gmv_all.append(gmv_item)
    align_bn_gmv_all.append(bn_gmv_item)
    align_cc_200_gmv_all.append(cc_200_gmv_item)
    align_basc_235_gmv_all.append(basc_235_gmv_item)
    align_y_all.append(y_item)

align_fnc_all = np.stack(align_fnc_all)
align_tc_all = np.stack(align_tc_all)
align_gmv_all = np.stack(align_gmv_all)
align_bn_gmv_all = np.stack(align_bn_gmv_all)
align_cc_200_gmv_all = np.stack(align_cc_200_gmv_all)
align_basc_235_gmv_all = np.stack(align_basc_235_gmv_all)
align_y_all = np.stack(align_y_all)
align_combine_aal_bn_cc_basc_gmv_all = np.concatenate((align_gmv_all, align_bn_gmv_all,align_cc_200_gmv_all,align_basc_235_gmv_all), axis=1)

data_dict = {
    'fnc': align_fnc_all,
    'tc': align_tc_all,
    'aal_116_gmv': align_gmv_all,
    'bn_274_gmv': align_bn_gmv_all,
    'cc_200_gmv': align_cc_200_gmv_all,
    'basc_235_gmv': align_basc_235_gmv_all,
    'combine_aal_bn_cc_basc_gmv': align_combine_aal_bn_cc_basc_gmv_all,
    'label': align_y_all
}

savemat(output_m, data_dict)