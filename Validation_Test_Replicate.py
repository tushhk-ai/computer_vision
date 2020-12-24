#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 15:31:00 2020

@author: tusharkalyani
"""
import os
import shutil

val_test_original = '/home/tusharkalyani/UPMC/UPMC_Kneecaps_Oct/Normal_rev_01_06_test/'
new_image_path = '/home/tusharkalyani/UPMC/UPMC_Kneecaps_Oct/new_cus_bff_2/'
val_test_new = '/home/tusharkalyani/UPMC/UPMC_Kneecaps_Oct/categorized/buff_35_40/'
for sub_folder in os.listdir(val_test_original):
    sub_folder_path = os.path.join(val_test_original,sub_folder)
    sub_folder_list = os.listdir(sub_folder_path)
    for all_imgs in sub_folder_list:
        for new_img in os.listdir(new_image_path):
            new_imgs_path = os.path.join(new_image_path,new_img)
            if new_img == all_imgs:
                des_path = os.path.join(val_test_new,sub_folder)
                if not os.path.exists(des_path):
                    os.makedirs(des_path, exist_ok=True)
                shutil.move(new_imgs_path, des_path)
                

    