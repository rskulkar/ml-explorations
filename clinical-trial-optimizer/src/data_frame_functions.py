import pandas as pd 

# ===== SOC: Format the DataFrame for the agent =====
def formatSOCLotDF(df):
    # Create a copy with percentages properly formatted
    df_formatted = df.copy()

    # Convert the fraction column to percentage strings
    if '% (Per Line)' in df_formatted.columns:
        df_formatted['% (Per Line)'] = (df_formatted['% (Per Line)'] * 100).round(2).astype(str) + '%'
    elif '% (per line)' in lot_df_formatted.columns:  # case-insensitive check
        df_formatted['% (per line)'] = (df_formatted['% (per line)'] * 100).round(2).astype(str) + '%'

    # Convert to markdown table for clean parsing
    df_string = df_formatted.to_markdown(index=False)
    
    return df_string

# ===== SOC: Filter LOT data for either a pct or Top N Regimens per line by Patient Share =====
def filterSOCLOT(df, pct, N):
    
    ## Take out top line patient count number 
    lot_df1 = df[~pd.isna(df['% (Per Line)'])]
    
    ## Extract L1 from L1R1, etc. 
    lot_df1['LineRegimen'] = ''
    lot_df1.loc[:, 'LineRegimen'] = lot_df1['Line Number'].astype('string').str.slice(0,2)
    
    ## Filter LOT data for Regimens with patient share > pct
    filtered_rows = lot_df1[lot_df1['% (Per Line)'] > pct].copy()
    
    ## Filter LOT data with Top N Regimens by patient share 
    topN_per_line = (lot_df1
                     .sort_values(['LineRegimen', '% (Per Line)'], ascending=[True, False])
                     .groupby('LineRegimen', group_keys=False)
                     .head(N)
                     .reset_index(drop=True)
                    )
    
    filtered_rows.drop(columns = 'LineRegimen', inplace=True)
    topN_per_line.drop(columns = 'LineRegimen', inplace=True)
    
    return filtered_rows, topN_per_line

# ===== ET: Format the DataFrame for the agent =====
def formatETLotDF(df):
    df_formatted = df.copy()
    
    # Convert to percentage strings
    if 'Period 1 % (Per Line)' in df_formatted.columns:
        df_formatted['Period 1 % (Per Line)'] = (df_formatted['Period 1 % (Per Line)'] * 100).round(2).astype(str) + '%'
    if 'Period 2 % (Per Line)' in df_formatted.columns:
        df_formatted['Period 2 % (Per Line)'] = (df_formatted['Period 2 % (Per Line)'] * 100).round(2).astype(str) + '%'
    if 'delta' in df_formatted.columns:
        df_formatted['delta'] = (df_formatted['delta'] * 100).round(2).astype(str) + '%'
    
    df_string = df_formatted.to_markdown(index=False)
    return df_string

# ===== ET: Assign Gap and Trend to LOT data =====
def filterETLOT(df1, df2 ):
    import numpy as np 
    import pandas as pd 
    from pandas.api.types import CategoricalDtype
    
    ## Match Regimens 
    df1.Regimen = df1.Regimen.str.lower()
    df2.Regimen = df2.Regimen.str.lower()
    
    ## Outer join of period 1 and period 2 data 
    df = pd.merge(df1, df2, on = ['Regimen','Line Number'], how = 'outer')
    df.loc[:, 'LineRegimen'] = df['Line Number'].astype('string').str.slice(0,2)
    df.columns = ['Regimen', 'Line Number', 'Period 1 Count', 'Period 1 % (Per Line)', 'Period 2Count','Period 2 % (Per Line)', 'LineRegimen']
    df = df[~(df.Regimen == 'total')] ## Remove head line data 
    
    ## Compute delta between periods 
        # Compute delta as Period2 - Period1, treating missing periods as 0 so we can capture changes
        # If both Period 1 and Period 2 are missing, keep delta as NA
    df['delta'] = df['Period 2 % (Per Line)'].fillna(0) - df['Period 1 % (Per Line)'].fillna(0)
    both_na = df['Period 2 % (Per Line)'].isna() & df['Period 1 % (Per Line)'].isna()
    df.loc[both_na, 'delta'] = pd.NA
    
    ## Assign GAP to Regimens 
    p1 = df['Period 1 % (Per Line)'].fillna(0)
    p2 = df['Period 2 % (Per Line)'].fillna(0)
    delta = df['delta'].fillna(0)

    conditions = [
        p2 >= 0.10,                         # Major SOC in period 2, even if delta >= 3%                   -> MAJOR
        (p2 >= 0.03) & (delta >= 0.03),     # Increasing by over 3%, but Minor -> Emerging SOC in period 2 -> EMERGING 
        (p2 >= 0.03) & (delta < 0.03),      # Minor SOC in period 2 with minor increase                    -> MINOR
        (p1 >= 0.03) & (delta < 0)          # (Minor/Major) SOC in Period 1, decreasing utilisation        -> MINOR
    ]
    choices = ['Major', 'Emerging', 'Minor','Minor']
    df['Gap Type'] = np.select(conditions, choices, default='No Gap')
    
    ## Assign Trend To Regimens 
    df['Trend'] = 'No Gap'
    mask_major = df['Gap Type'] == 'Major'
    mask_minor = df['Gap Type'] == 'Minor'

    # Major
    df.loc[mask_major & (df['delta'] >= 0.03), 'Trend'] = 'Increasing'
    df.loc[mask_major & (df['delta'] < -0.03), 'Trend'] = 'Decreasing'
    df.loc[mask_major & (df['delta'] >= -0.03) & (df['delta'] < 0.03), 'Trend'] = 'Stable'

    # Minor
    df.loc[mask_minor & (df['delta'] >= 0.03), 'Trend'] = 'Emerging'
    df.loc[mask_minor & (df['delta'] < -0.03), 'Trend'] = 'Decreasing'
    df.loc[mask_minor & (df['delta'] >= -0.03) & (df['delta'] < 0.03), 'Trend'] = 'Stable'

    # For Gap Type == Emerging, Trend is set to 'No Gap', needs to be changed 
    df.loc[df['Gap Type'] == 'Emerging', 'Trend'] = df.loc[df['Gap Type'] == 'Emerging', 'Trend'].where(df['delta'] < 0.03, 'Emerging')
    
    ## Sort Gap + Trend by highest delta
    # Define logical sort order
    gap_order = CategoricalDtype(['Emerging', 'Major', 'Minor', 'No Gap'], ordered=True)
    trend_order = CategoricalDtype(['Emerging', 'Increasing', 'Decreasing', 'Stable', 'No Gap'], ordered=True)

    df['Gap Type'] = df['Gap Type'].astype(gap_order)
    df['Trend'] = df['Trend'].astype(trend_order)
    
    df = df.sort_values(['Gap Type','Trend','delta'], ascending=[True,True,False], ignore_index=True)
    
    ## Drop No Gap + No Gap (Trend) rows
    df = df[~((df['Gap Type'] == 'No Gap') & (df['Trend'] == 'No Gap'))]
    
    ## Drop columns 
    df.drop(columns = {'Period 1 Count', 'Period 2Count', 'LineRegimen'}, inplace=True)
    
    return df