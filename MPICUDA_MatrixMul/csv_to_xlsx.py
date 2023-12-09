import pandas as pd

# Read the CSV file
df = pd.read_csv('output.csv', header=None)

# Specify the Excel file name
excel_file_name = 'output.xlsx'

# Write the DataFrame to an Excel file
df.to_excel(excel_file_name, index=False, header=False)