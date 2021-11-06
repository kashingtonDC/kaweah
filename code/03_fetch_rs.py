import os
import ee
import datetime
import time
import sklearn
import importlib

import geopandas as gp
import pandas as pd
import numpy as np
import rsfuncs as rs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pandas.tseries.offsets import MonthEnd
from dateutil.relativedelta import relativedelta
from sklearn import preprocessing

from tqdm import tqdm_notebook as tqdm

ee.Initialize()


# Load shapefile 
shp = gp.read_file("../shape/kb_rpj.shp")

# Make EE objects from shapefiles 
area = rs.gdf_to_ee_poly(shp)

data = rs.load_data()

strstart = '2001-01-01'
strend = '2020-12-30'

startdate = datetime.datetime.strptime(strstart, "%Y-%m-%d")
enddate = datetime.datetime.strptime(strend, "%Y-%m-%d")

# Precip
gpm = rs.calc_monthly_sum(data['gpm'], startdate, enddate, area)
prism = rs.calc_monthly_sum(data['prism'], startdate, enddate, area)
dmet = rs.calc_monthly_sum(data['dmet'], startdate, enddate, area)
chirps = rs.calc_monthly_sum(data['chirps'], startdate, enddate, area)
psn = rs.calc_monthly_sum(data['persiann'], startdate, enddate, area)

# Aet
modis_aet = rs.calc_monthly_sum(data['modis_aet'], startdate, enddate, area)
gldas_aet = rs.calc_monthly_sum(data['gldas_aet'], startdate, enddate, area)
tc_aet = rs.calc_monthly_sum(data['tc_aet'], startdate, enddate, area)
fldas_aet = rs.calc_monthly_sum(data['fldas_aet'], startdate, enddate, area)

# PET
gldas_pet = rs.calc_monthly_sum(data['gldas_pet'], startdate, enddate, area)
modis_pet = rs.calc_monthly_sum(data['modis_pet'], startdate, enddate, area)
nldas_pet = rs.calc_monthly_sum(data['nldas_pet'], startdate, enddate, area)
tc_pet = rs.calc_monthly_sum(data['tc_pet'], startdate, enddate, area)
gmet_eto = rs.calc_monthly_sum(data['gmet_eto'], startdate, enddate, area)

# SM

# SMOS 
smos_ssm = rs.calc_monthly_mean(data['smos_ssm'], "2010-01-01", enddate, area)
smos_susm = rs.calc_monthly_mean(data['smos_susm'],"2010-01-01", enddate, area)
smos_smp = rs.calc_monthly_mean(data['smos_smp'],"2010-01-01", enddate, area)
smos_sm = pd.concat([smos_ssm, smos_susm], axis = 1).sum(axis =1)

# TC
tc_sm = rs.calc_monthly_mean(data['tc_sm'], startdate, enddate, area)

# GLDAS
gldas_rzsm = rs.calc_monthly_mean(data['gldas_rzsm'], startdate, enddate, area)
gldas_gsm1 = rs.calc_monthly_mean(data['gsm1'], startdate, enddate, area)
gldas_gsm2 = rs.calc_monthly_mean(data['gsm2'], startdate, enddate, area)
gldas_gsm3 = rs.calc_monthly_mean(data['gsm3'], startdate, enddate, area)
gldas_gsm4 = rs.calc_monthly_mean(data['gsm4'], startdate, enddate, area)
gldas_sm = pd.concat([gldas_gsm1,gldas_gsm2,gldas_gsm3,gldas_gsm4], axis = 1).sum(axis =1)

# FLDAS 
fldas_fsm1 = rs.calc_monthly_mean(data['fsm1'], startdate, enddate, area_cv)
fldas_fsm2 = rs.calc_monthly_mean(data['fsm2'], startdate, enddate, area_cv)
fldas_fsm3 = rs.calc_monthly_mean(data['fsm3'], startdate, enddate, area_cv)
fldas_fsm4 = rs.calc_monthly_mean(data['fsm4'], startdate, enddate, area_cv)
fldas_sm = pd.concat([fldas_fsm1,fldas_fsm2,fldas_fsm3,fldas_fsm4], axis = 1).sum(axis =1)

# SMAP
smap_ssm = rs.calc_monthly_mean(data['smap_ssm'], '2015-04-01', enddate, area)
smap_susm = rs.calc_monthly_mean(data['smap_susm'],'2015-04-01', enddate, area)
smap_smp = rs.calc_monthly_mean(data['smap_smp'],'2015-04-01', enddate, area)
smap_sm = pd.concat([smap_ssm, smap_susm], axis = 1).sum(axis =1)


# Merge 
gldas_sm = pd.DataFrame(pd.concat([gldas_gsm1,gldas_gsm2,gldas_gsm3,gldas_gsm4], axis = 1).sum(axis =1))
gldas_sm.columns = ['gldas_sm']
fldas_sm = pd.DataFrame(pd.concat([fldas_fsm1,fldas_fsm2,fldas_fsm3,fldas_fsm4], axis = 1).sum(axis =1))
fldas_sm.columns = ['fldas_sm']
smap_sm = pd.DataFrame(pd.concat([smap_ssm, smap_susm], axis = 1).sum(axis =1))
smap_sm.columns = ['smap_sm']
smos_sm = pd.DataFrame(pd.concat([smos_ssm, smos_susm], axis = 1).sum(axis =1))
smos_sm.columns = ['smos_sm']

# SWe
gldas_swe = rs.calc_monthly_mean(data['gldas_swe'], startdate, enddate, area)
fldas_swe = rs.calc_monthly_mean(data['fldas_swe'],startdate, enddate, area)
dmet_swe = rs.calc_monthly_mean(data['dmet_swe'],startdate, enddate, area)
tc_swe = rs.calc_monthly_mean(data['tc_swe'],startdate, enddate, area)

pdfs = {"p_prism":prism, "p_gpm":gpm, "p_dmet":dmet, "p_chirps": chirps, "p_psn":psn}
aetdfs = {"aet_modis":modis_aet, "aet_gldas":gldas_aet, "aet_tc":tc_aet, "aet_fldas":fldas_aet }
petdfs = {"pet_modis":modis_pet, "pet_gldas":gldas_pet, "pet_tc":tc_pet, "pet_nldas":nldas_pet, 'pet_gmet':gmet_eto }
smdfs = {"sm_smos": smos_sm, "sm_smap": smap_sm, "sm_tc": tc_sm, "sm_gldas": gldas_sm }
swedfs = {'swe_gldas': gldas_swe, 'swe_fldas': fldas_swe, 'swe_dmet':dmet_swe, "swe_tc":tc_swe}


master_df = []
for i in [pdfs, aetdfs,petdfs, smdfs, swedfs]:
    for k,v in i.items():
        print(k,v.columns)
        newdf = v
        newdf.columns = [k + "_cvws"] 
        master_df.append(newdf)

finout = pd.concat(master_df, axis = 1)

finout.to_csv('../data/RS_analysis_dat_cvws.csv')


