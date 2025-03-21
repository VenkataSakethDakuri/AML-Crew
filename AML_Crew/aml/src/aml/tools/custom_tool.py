from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")




class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."


class PredictorToolInput(BaseModel):
    """Input schema for Predictor Tool."""
    pass

class PredictorTool(BaseTool):
    name: str = "Predictor Tool"
    description: str = (
        "This tool predicts whether a transaction is fraudulent or not."
    )
    args_schema: Type[BaseModel] = PredictorToolInput

    def _run(self) -> None:

        import pandas as pd
        import joblib

        # Load the trained model and transformer
        model = joblib.load('xgboost_model.joblib')

        sample_data = {
            'From Bank': [70],
            'Account': ['100428660'],
            'To Bank': [1124],
            'Account.1': ['800825340'],
            'Amount Received': [389769.39],
            'Receiving Currency': ['US Dollar'],
            'Amount Paid': [389769.39],
            'Payment Currency': ['US Dollar'],
            'Payment Format': ['Cheque'],
            'Date': ['2022-09-01'],
            'Day': ['Thursday'],
            'Time': ['00:21:00']
        }


        # Convert sample data to a DataFrame
        sample_df = pd.DataFrame(sample_data)

        # Predict using the trained model
        prediction = model.predict(sample_df)

        # Output the result
        print(f"Prediction: {prediction[0]}")  # 'prediction[0]' will give the class (0 or 1)



class ReportToolInput(BaseModel):
    """Input schema for Report Tool."""
    pass


class ReportTool(BaseTool):
    name = "Report Tool"
    description = "This tool analyzes a transaction and creates a report."
    args_schema : Type[BaseModel] = ReportToolInput
    def _run(self) -> None:
        import pandas as pd

        from aml.src.aml.explainer import analyze_transaction_and_create_report

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

