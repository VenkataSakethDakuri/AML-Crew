# Example usage
import joblib
import pandas as pd

from explainer import analyze_transaction_and_create_report

# Example transaction - removed nested lists
sample_data = {
    'From Bank': 70,
    'Account': '100428660',
    'To Bank': 1124,
    'Account.1': '800825340',
    'Amount Received': 389769.39,
    'Receiving Currency': 'US Dollar',
    'Amount Paid': 389769.39,
    'Payment Currency': 'US Dollar',
    'Payment Format': 'Cheque',
    'Date': '2022-09-01',
    'Day': 'Thursday',
    'Time': '00:21:00'
}

# Convert sample data to DataFrame - wrap in list for single row
sample_df = pd.DataFrame([sample_data])
    
report_path = analyze_transaction_and_create_report(
    sample_df, 
    output_dir='./reports'      # Where to save reports
)

print(f"Analysis complete. Report saved to: {report_path}")