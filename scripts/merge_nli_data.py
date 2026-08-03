import pandas as pd
import numpy as np

# FEVER
fever = pd.read_parquet('data/processed/fact_verification/fever_clean.parquet')[['claim','label']]
fever['label'] = fever['label'].map({'True':'SUPPORTS','False':'REFUTES','Unverified':'NEI'})

# FEVEROUS — undersample to fix severe imbalance
fev = pd.read_parquet('data/processed/fact_verification/feverous_clean.parquet')[['claim','label']]
fev['label'] = fev['label'].map({'True':'SUPPORTS','False':'REFUTES','Unverified':'NEI'})
fev_sup = fev[fev['label']=='SUPPORTS']  # 2232 rows, keep all
fev_ref = fev[fev['label']=='REFUTES'].sample(6000, random_state=42)
fev_nei = fev[fev['label']=='NEI'].sample(6000, random_state=42)
fev_balanced = pd.concat([fev_sup, fev_ref, fev_nei])

# LIAR
liar = pd.read_parquet('data/processed/fact_verification/liar_clean.parquet')[['claim','label']]
liar['label'] = liar['label'].map({'True':'SUPPORTS','False':'REFUTES','Misleading':'NEI'})

combined = pd.concat([fever, fev_balanced, liar], ignore_index=True).dropna(subset=['label'])
print(combined['label'].value_counts())
print('Total rows:', len(combined))
combined.to_parquet('data/processed/fact_verification/nli_combined.parquet', index=False)
print('Saved.')
