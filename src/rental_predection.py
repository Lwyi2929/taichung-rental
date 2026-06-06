from __future__ import annotations

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay
import xgboost as xgb
import lightgbm as lgb
import libpysal
import time, os, gc

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = APP_ROOT / "df_poi_3.gpkg"
POI_DIR = APP_ROOT / "POIs"
ARTIFACT_DIR = APP_ROOT / "artifacts"
MODEL_ARTIFACT_PATH = ARTIFACT_DIR / "lightgbm_rent_model.joblib"
SPATIAL_CACHE_PATH = ARTIFACT_DIR / "spatial_cache.joblib"

EPSG_MODEL = 3826
EPSG_WEB = 4326
SEED = 42
ML_N = 30_000
KNN_K = 20

gdf = gpd.read_file(DATA_PATH)

if '建物型態' in gdf.columns:
    gdf['建物型態'] = gdf['建物型態'].astype(str).str.replace(' ', '')
    gdf = pd.get_dummies(gdf, columns=['建物型態'], drop_first=True, dtype=float)
# ================= 整理好的：屋齡轉換區塊 =================
gdf['交易年月日'] = pd.to_numeric(gdf['交易年月日'], errors='coerce').fillna(0)
gdf['建築完成年月'] = pd.to_numeric(gdf['建築完成年月'], errors='coerce').fillna(0)
gdf['交易年'] = (gdf['交易年月日'] // 10000).astype(int)
gdf['建築年'] = (gdf['建築完成年月'] // 10000).astype(int)
gdf['屋齡'] = gdf['交易年'] - gdf['建築年']
gdf.loc[gdf['屋齡'] < 0, '屋齡'] = 0
gdf.loc[gdf['屋齡'] > 100, '屋齡'] = 0
# ===============================================================
building_dummies = [col for col in gdf.columns if col.startswith('建物型態_')]

gdf['inter_apt']   = gdf['dist_to_mrt_log'] * gdf['建物型態_公寓(5樓含以下無電梯)']
gdf['inter_university']  = gdf['dist_to_mrt_log'] * gdf['dist_to_university_log']
gdf['inter_core']  = gdf['dist_to_mrt_log'] * gdf['is_core']

BASE_X = [
    'dist_to_mrt_log', 'dist_to_social_log', 'dist_to_train_log', 'dist_to_hosp_log',
    'dist_to_park_log', 'dist_to_university_log', 'straight_dist_nimby_log',
    'straight_dist_temple_log', 'dist_to_attraction_log', 'elem_count_3km_log',
    'store_count_500m_log', '屋齡','is_core'
]+ building_dummies

CONTINUOUS_VARS = [
    'dist_to_mrt_log', 'dist_to_social_log', 'dist_to_train_log', 'dist_to_hosp_log',
    'dist_to_park_log', 'dist_to_university_log', 'straight_dist_nimby_log',
    'straight_dist_temple_log', 'dist_to_attraction_log'
]
scaler = StandardScaler()
gdf_s = gdf.copy()
gdf_s[CONTINUOUS_VARS] = scaler.fit_transform(gdf[CONTINUOUS_VARS])
gdf_s['geometry'] = gdf.geometry.apply(
    lambda geom: Point(geom.x + np.random.normal(0, 0.01),
                       geom.y + np.random.normal(0, 0.01))
) #空間擾動
t0 = time.time()
w = libpysal.weights.KNN.from_dataframe(gdf_s, k=20) #空間權重矩陣knn取20點
w.transform = 'r'
print(f'KNN-20 完成 {time.time()-t0:.1f}s')

y_arr = gdf_s['單價元平方公尺'].values
gdf_s['Wy']             = libpysal.weights.lag_spatial(w, y_arr) #做空間滯後項
gdf_s['W_dist_to_mrt_log'] = libpysal.weights.lag_spatial(w, gdf_s['dist_to_mrt_log'].values)


USER_CONTINUOUS = ["area_pings", "deposit_months", "mgmt_fee", "water_fee", "sum_equip_idx"]
USER_BINARY = [
    "pet_friendly", "limited", "parking", "apartment", "elevator_building",
    "air_conditioner", "laundry",
]

POI_LAYERS = {
    "railsta": (POI_DIR / "Taichung_rail_stations.gpkg", "rail_station"),
    "mrt": (POI_DIR / "Taichung_MRT.gpkg", "Taichung_MRT"),
    "ubike": (POI_DIR / "Taichung_youbikes.gpkg", "youbike20"),
    "highway": (POI_DIR / "Taichung_highway_inters.gpkg", "taichung_highway_inters"),
    "park": (POI_DIR / "Taichung_parks.gpkg", "parks"),
    "school": (POI_DIR / "Taichung_schools.gpkg", "schools"),
    "temple": (POI_DIR / "Taichung_temples.gpkg", "temples"),
    "stores": (POI_DIR / "Taichung_stores.gpkg", "stores"),
    "busstops": (POI_DIR / "Taichung_busstops.gpkg", "busstops"),
    "medical": (POI_DIR / "Taichung_medical_service.gpkg", "hospital_done_0924"),
    "towns": (POI_DIR / "taichung_town_joined_2.gpkg", "taichung_town_joined_2"),
    "roads": (POI_DIR / "112Taichung_road_network.gpkg", "112Taichung_road_network"),
}

ROAD_DISTANCE_FEATURES = {
    "railsta": "ln_dist_road_railsta",
    "mrt": "ln_dist_road_mrt",
    "ubike": "ln_dist_road_ubike",
    "highway": "ln_dist_road_highway",
    "park": "ln_dist_road_park",
    "school": "ln_dist_road_school",
}

CORE_TOWNS = {"東區", "西區", "南區", "北區", "中區", "西屯區", "北屯區", "南屯區"}

FEATURE_DESCRIPTIONS = {
    "pet_friendly": "是否可養寵物，1=可，0=不可。",
    "limited": "租屋限制條件，1=有限制，0=未標示限制。",
    "deposit_months": "押金月數。",
    "mgmt_fee": "管理費。",
    "water_fee": "水費。",
    "area_pings": "出租坪數。",
    "parking": "是否提供停車位。",
    "apartment": "是否為公寓類型。",
    "elevator_building": "是否為電梯大樓。",
    "ln_dist_road_railsta": "至最近火車站的道路距離取自然對數。",
    "ln_dist_road_mrt": "至最近捷運站出口的道路距離取自然對數。",
    "ln_dist_road_ubike": "至最近 YouBike 站的道路距離取自然對數。",
    "ln_dist_road_highway": "至最近交流道的道路距離取自然對數。",
    "ln_dist_road_park": "至最近公園的道路距離取自然對數。",
    "ln_dist_road_school": "至最近學校的道路距離取自然對數。",
    "ln_dist_eucl_temple": "至最近寺廟的直線距離取自然對數。",
    "ln_stores_500m": "500 公尺環域內商店數量取 ln(count+1)。",
    "ln_bus_stops_500m": "500 公尺環域內公車站數量取 ln(count+1)。",
    "ln_medical_service_500m": "500 公尺環域內醫療診所數量取 ln(count+1)。",
    "core_zone": "是否位於核心商業區。",
    "air_conditioner": "是否提供冷氣。",
    "laundry": "是否提供洗衣設備。",
    "sum_equip_idx": "家具、其他家電與影音網路設備指標總和。",
    "inter_equip": "pet_friendly 與 sum_equip_idx 的交互項。",
    "inter_apt": "pet_friendly 與 apartment 的交互項。",
    "inter_elev": "pet_friendly 與 elevator_building 的交互項。",
    "inter_core": "pet_friendly 與 core_zone 的交互項。",
    "Wy": "目標點鄰近 20 筆租屋樣本 ln_rent 的空間滯後平均。",
    "W_pet_friendly": "目標點鄰近 20 筆租屋樣本 pet_friendly 的空間滯後平均。",
}
