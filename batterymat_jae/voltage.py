import pandas as pd

# Read the CSV
df = pd.read_csv('summary.csv')

# Filter for Success status
df_success = df[df['status'] == 'Success'].copy()

# Convert avg_voltage to numeric (converts 'N/A' to NaN automatically)
df_success['avg_voltage_V'] = pd.to_numeric(df_success['avg_voltage_V'], errors='coerce')

# Drop rows with NaN voltage
df_success = df_success.dropna(subset=['avg_voltage_V'])

# Create output dataframe
output = df_success[['jid', 'avg_voltage_V']].copy()

# Save WITHOUT header (ALIGNN CSV format expects no header)
output.to_csv('id_prop.csv', index=False, header=False)

print(f"✓ Created id_prop.csv with {len(output)} materials")
print(f"\nVoltage statistics:")
voltages = output['avg_voltage_V'].values
print(f"  Count: {len(voltages)}")
print(f"  Min: {voltages.min():.3f} V")
print(f"  Max: {voltages.max():.3f} V")
print(f"  Mean: {voltages.mean():.3f} V")
