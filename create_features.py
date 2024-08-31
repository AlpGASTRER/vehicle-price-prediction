import pandas as pd
import json

# Load your CSV file
df = pd.read_csv('final_df.csv')

# Create the feature data dictionary
feature_data = {
    'year': sorted(df['year'].unique().tolist()),
    'make': sorted(df['make'].unique().tolist()),
    'model_by_make': {},
    'trim_by_make_model': {},
    'body': sorted(df['body'].unique().tolist()),
    'transmission': sorted(df['transmission'].unique().tolist()),
    'state': sorted(df['state'].unique().tolist()),
    'condition': list(range(1, 50)),  # Assuming condition is 1-49
    'color': sorted(df['color'].unique().tolist()),
    'interior': sorted(df['interior'].unique().tolist()),
    'seller': sorted(df['seller'].unique().tolist()),
    'season': sorted(df['season'].unique().tolist()),
}

# Populate model_by_make
for make in feature_data['make']:
    feature_data['model_by_make'][make] = sorted(df[df['make'] == make]['model'].unique().tolist())

# Populate trim_by_make_model
for make in feature_data['make']:
    for model in feature_data['model_by_make'][make]:
        key = f"{make}_{model}"
        feature_data['trim_by_make_model'][key] = sorted(
            df[(df['make'] == make) & (df['model'] == model)]['trim'].unique().tolist()
        )

# Save to JSON file
with open('feature_data.json', 'w') as f:
    json.dump(feature_data, f)

print("feature_data.json has been created successfully.")