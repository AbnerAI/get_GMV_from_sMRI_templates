import csv
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd

# 加载sMRI数据和AAL模板
sMRI_path = '/mnt/data/home/cxx/PycharmProjects/pythonProject/Multi-modal/end-to-end-personalized_templates/pytorch_version/data/973_sMRI'
# AAL_1_5mm.nii, BN_Atlas_274_with_cerebellum_without_255_1_5mm.nii
# rADHD200_parcellate_200.nii, rtemplate_cambridge_basc_multiscale_asym_scale325.nii
aal_template_path = '/mnt/data/home/cxx/PycharmProjects/pythonProject/Multi-modal/end-to-end-personalized_templates/pytorch_version/data/Atlas/Original/rADHD200_parcellate_200.nii'#
all_sMRI = os.listdir(sMRI_path)
output_path = f"{os.path.basename(aal_template_path)[:-4]}_Output_Region_GMV.csv"
# setting columns
columns_list = ['Name']

for idx, file_name in enumerate(all_sMRI):
    print('processing: ', file_name)
    sys.stdout.flush()
    sMRI_img = nib.load(os.path.join(sMRI_path, file_name))
    aal_img = nib.load(aal_template_path)

    gm_data = sMRI_img.get_fdata()
    # gm_data = np.nan_to_num(sMRI_data, nan=0)
    
    gm_nan_mask = ~np.isnan(gm_data)
    gm_data_nan_mask = gm_data.copy()
    gm_data_nan_mask[gm_nan_mask] = 1
    gm_data_nan_mask[gm_data_nan_mask!=1] = 0

    # remove Nan
    aal_data = aal_img.get_fdata()
    aal_data = np.nan_to_num(aal_data)
    aal_data = aal_data * gm_data_nan_mask

    # 获取体素大小
    voxel_volume = np.prod(aal_img.header.get_zooms())

    # 遍历AAL模板中的每一个区域
    all_average_gmv = []
    all_average_gmv.append(file_name)
    for region_id in np.unique(aal_data):
        if region_id == 0: # 跳过背景区域
            continue

        # 创建二值掩模
        mask = (aal_data == region_id).astype(int)

        # 假设mask是脑区1的二值掩模，gm_data是灰质数据
        gm_data = np.nan_to_num(gm_data) # 0*nan = nan, so ....
        roi_gm_data = gm_data * mask

        # 计算掩模内灰质体素的总和
        total_gm_value = np.sum(roi_gm_data)

        # 计算掩模内体素的数量
        num_voxels = np.sum(mask)

        # 计算平均灰质密度
        average_gm_density = total_gm_value / num_voxels

        # 这个平均灰质密度就是你所谓的平均GMV
        average_gmv = average_gm_density
        all_average_gmv.append(average_gmv)
        # 存储结果
        columns_list.append(f"{os.path.basename(aal_template_path)[:-4]}_Region_{int(region_id)}")
        # roi_volumes.loc[len(roi_volumes)] = [f"Region_{region_id}", average_gmv]
    if idx == 0:
        with open(output_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(columns_list)
            writer.writerow(all_average_gmv)
    else:
        with open(output_path, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(all_average_gmv)
