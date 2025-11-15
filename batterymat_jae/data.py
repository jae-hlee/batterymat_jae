import pandas as pd
import json
import pickle
import os
from jarvis.db.figshare import data

# Cache file
CACHE_FILE = 'jarvis_dft3d_cache.pkl'
if os.path.exists(CACHE_FILE):
    print("Loading cached JARVIS database...")
    with open(CACHE_FILE, 'rb') as f:
        dft3d = pickle.load(f)
else:
    print("Downloading JARVIS database (one-time, ~5 min)...")
    dft3d = data('dft_3d')
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(dft3d, f)

print(f"Database loaded: {len(dft3d)} total structures")

# Create lookup dictionary
jid_lookup = {entry['jid']: entry for entry in dft3d}

# Read your CSV
df = pd.read_csv('id_prop.csv', header=None, names=['jid', 'voltage'])

print(f"Creating dataset for {len(df)} materials...")

dataset = []
not_found = []

for _, row in df.iterrows():
    jid = row['jid']
    voltage = row['voltage']
    
    if jid in jid_lookup:
        entry = jid_lookup[jid]
        dataset.append({
            'jid': jid,
            'atoms': entry['atoms'],
            'target': float(voltage)
        })
    else:
        not_found.append(jid)

# Save
with open('id_prop.json', 'w') as f:
    json.dump(dataset, f)

print(f"\n✓ Created id_prop.json with {len(dataset)} materials")
if not_found:
    print(f"✗ Not found in database: {len(not_found)} JIDs")
