import pandas as pd

# Read the CSV files into separate DataFrames
df_std = pd.read_csv('default_representation_std.csv')
df_train_loss = pd.read_csv('default_train_loss.csv')
df_kNN_accuracy = pd.read_csv('default_kNN_accuracy.csv')

# Resample the 'step' column to a common interval of 10
df_std_resampled = df_std.resample('10', on='step').mean()
df_train_loss_resampled = df_train_loss.resample('10', on='step').mean()
df_kNN_accuracy_resampled = df_kNN_accuracy.resample('10', on='step').mean()

# Merge the resampled DataFrames based on the 'step' column
merged_df = df_std_resampled.merge(df_train_loss_resampled, on='step').merge(df_kNN_accuracy_resampled, on='step')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_table.csv', index=False)
