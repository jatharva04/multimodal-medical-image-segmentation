import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.stats.multitest import multipletests
import warnings
import scipy.stats as st
warnings.filterwarnings('ignore')

data_2d = pd.read_csv("2d_metrics.csv")

data_3d = pd.read_csv("3d_metrics.csv")
results = []
raw_p_values = []
valid_indices = []

# 2. Loop through each organ to run Wilcoxon, Effect Size, and CI
for i, organ in enumerate(data_2d.columns):
    scores_2d = np.array(data_2d[organ])
    scores_3d = np.array(data_3d[organ])
    
    valid_mask = ~np.isnan(scores_2d) & ~np.isnan(scores_3d)
    valid_2d = scores_2d[valid_mask]
    valid_3d = scores_3d[valid_mask]
    
    mean_2d = round(np.nanmean(scores_2d), 4) if not np.isnan(scores_2d).all() else "N/A"
    mean_3d = round(np.nanmean(scores_3d), 4) if not np.isnan(scores_3d).all() else "N/A"
    
    p_value = np.nan
    eff_size = np.nan
    ci_str = "N/A"
    
    if len(valid_2d) > 0:
        differences = valid_2d - valid_3d
        
        # --- CONFIDENCE INTERVAL CALCULATION ---
        if len(differences) > 1 and np.std(differences) > 0:
            ci = st.t.interval(0.95, df=len(differences)-1, loc=np.mean(differences), scale=st.sem(differences))
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        elif np.all(differences == 0):
            ci_str = "[0.0000, 0.0000]"
            
        if not np.all(differences == 0):
            res = pg.wilcoxon(valid_2d, valid_3d)
            
            # Dynamic p-value
            if 'p-val' in res.columns: p_value = res['p-val'].values[0]
            elif 'p-unc' in res.columns: p_value = res['p-unc'].values[0]
            elif 'pval' in res.columns: p_value = res['pval'].values[0]
            else: p_value = res.iloc[0, 2] 

            # Dynamic Effect Size
            if 'RBC' in res.columns: eff_size = res['RBC'].values[0]
            elif 'CLES' in res.columns: eff_size = res['CLES'].values[0]
            else: eff_size = res.iloc[0, 3] 
            
            raw_p_values.append(p_value)
            valid_indices.append(i)
            
    results.append({
        'Organ': organ,
        'Mean 2D': mean_2d,
        'Mean 3D': mean_3d,
        'Effect_Size_RBC': round(eff_size, 4) if not np.isnan(eff_size) else "N/A",
        '95%_CI_Diff': ci_str, 
        'Adj_P_Value': np.nan 
    })

# 3. Apply Benjamini-Hochberg FDR Correction
_, p_adjusted, _, _ = multipletests(raw_p_values, method='fdr_bh')

for idx, adj_p in zip(valid_indices, p_adjusted):
    results[idx]['Adj_P_Value'] = round(adj_p, 4)

for res in results:
    adj_p = res['Adj_P_Value']
    if pd.isna(adj_p):
        res['Adj_P_Value'] = "N/A"
        res['Significance (FDR)'] = "N/A"
    else:
        res['Significance (FDR)'] = "Significant" if adj_p < 0.05 else "Not Significant"

results_df = pd.DataFrame(results)
results_df[['Organ', 'Mean 2D', 'Mean 3D', 'Effect_Size_RBC', '95%_CI_Diff', 'Adj_P_Value', 'Significance (FDR)']].to_csv("FDR_Corrected_Table3_with_CI.csv", index=False)