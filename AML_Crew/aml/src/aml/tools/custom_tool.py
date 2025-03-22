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
    from_bank: int = Field(..., description="Sender bank code")
    account: str = Field(..., description="Sender account number")
    to_bank: int = Field(..., description="Receiver bank code")
    account_dest: str = Field(..., description="Receiver account number")
    amount_received: float = Field(..., description="Amount received in transaction")
    receiving_currency: str = Field(..., description="Currency of the received amount")
    amount_paid: float = Field(..., description="Amount paid in transaction")
    payment_currency: str = Field(..., description="Currency of the payment")
    payment_format: str = Field(..., description="Format of payment (e.g., Cheque, Wire)")
    date: str = Field(..., description="Transaction date (YYYY-MM-DD)")
    day: str = Field(..., description="Day of the week")
    time: str = Field(..., description="Transaction time (HH:MM:SS)")

class PredictorTool(BaseTool):
    name: str = "Predictor Tool"
    description: str = (
        "This tool predicts whether a transaction is fraudulent or not based on transaction details."
    )
    args_schema: Type[BaseModel] = PredictorToolInput

    def _run(self, from_bank: int, account: str, to_bank: int, account_dest: str, 
             amount_received: float, receiving_currency: str, amount_paid: float, 
             payment_currency: str, payment_format: str, date: str, day: str, time: str) -> str:

        import pandas as pd
        import joblib
        import os

        # Create transaction data dictionary from input parameters
        transaction_data = {
            'From Bank': [from_bank],
            'Account': [account],
            'To Bank': [to_bank],
            'Account.1': [account_dest],
            'Amount Received': [amount_received],
            'Receiving Currency': [receiving_currency],
            'Amount Paid': [amount_paid],
            'Payment Currency': [payment_currency],
            'Payment Format': [payment_format],
            'Date': [date],
            'Day': [day],
            'Time': [time]
        }

        # Convert transaction data to a DataFrame
        transaction_df = pd.DataFrame(transaction_data)

        try:
            # Load the trained model
            model_path = 'xgboost_model.joblib'
            if not os.path.exists(model_path):
                model_path = os.path.join(os.path.dirname(__file__), '..', 'xgboost_model.joblib')
            
            model = joblib.load(model_path)

            # Predict using the trained model
            prediction = model.predict(transaction_df)
            prediction_proba = model.predict_proba(transaction_df)
            
            # Get confidence score
            confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
            confidence_percentage = round(confidence * 100, 2)
            
            # Format the result
            if prediction[0] == 1:
                result = {
                    "prediction": "Fraud",
                    "confidence": confidence_percentage,
                    "transaction_data": transaction_data
                }
            else:
                result = {
                    "prediction": "Not Fraud",
                    "confidence": confidence_percentage,
                    "transaction_data": transaction_data
                }
            
            return str(result)
            
        except Exception as e:
            return f"Error making prediction: {str(e)}"


class ReportToolInput(BaseModel):
    """Input schema for Report Tool."""
    prediction_result: str = Field(..., description="The prediction result from the Predictor Tool")


class ReportTool(BaseTool):
    name = "Report Tool"
    description = "This tool analyzes a transaction and creates a detailed report based on the prediction result."
    args_schema : Type[BaseModel] = ReportToolInput
    
    def _run(self, prediction_result: str) -> str:
        import pandas as pd
        import ast
        import os
        from aml.src.aml.explainer import analyze_transaction_and_create_report

        try:
            # Parse the prediction result string into a dictionary
            prediction_data = ast.literal_eval(prediction_result)
            
            # Extract transaction data
            transaction_data = prediction_data.get("transaction_data", {})
            
            # Convert the transaction data format from list values to single values
            flat_transaction_data = {}
            for key, value in transaction_data.items():
                flat_transaction_data[key] = value[0] if isinstance(value, list) and len(value) > 0 else value
            
            # Convert flat transaction data to DataFrame
            sample_df = pd.DataFrame([flat_transaction_data])
            
            # Create output directory if it doesn't exist
            output_dir = './reports'
            os.makedirs(output_dir, exist_ok=True)
            
            # Analyze transaction and create report
            report_path = analyze_transaction_and_create_report(
                sample_df, 
                output_dir=output_dir
            )
            
            # Return the path to the report
            return f"Report created successfully at {report_path}. Prediction: {prediction_data.get('prediction', 'Unknown')} with {prediction_data.get('confidence', 0)}% confidence."
            
        except Exception as e:
            return f"Error creating report: {str(e)}"


class CaseManagerToolInput(BaseModel):
    """Input schema for Case Manager Tool."""
    action: str = Field(..., description="Action to perform: create_case, update_status, prioritize, list_cases, get_case")
    case_id: str = Field(None, description="Case ID for the case to work with (for update_status, prioritize, get_case)")
    transaction_data: dict = Field(None, description="Transaction data for creating a new case")
    new_status: str = Field(None, description="New status for update_status action (new, assigned, in_progress, pending, resolved)")
    assignee: str = Field(None, description="Investigator assigned to the case")
    risk_factors: dict = Field(None, description="Risk factors associated with the case for prioritization")

class CaseManagerTool(BaseTool):
    name = "Case Manager Tool"
    description = "This tool manages AML investigation cases including creation, tracking, status updates, and risk-based prioritization."
    args_schema: Type[BaseModel] = CaseManagerToolInput
    
    def __init__(self):
        super().__init__()
        self.cases = {}  # In-memory case storage (would be replaced with database in production)
        self.case_counter = 1000  # Starting case ID counter
    
    def _run(self, action: str, case_id: str = None, transaction_data: dict = None, 
             new_status: str = None, assignee: str = None, risk_factors: dict = None) -> str:
        import json
        import uuid
        import datetime
        
        if action == "create_case":
            return self._create_case(transaction_data, assignee, risk_factors)
        elif action == "update_status":
            return self._update_case_status(case_id, new_status)
        elif action == "prioritize":
            return self._prioritize_case(case_id, risk_factors)
        elif action == "list_cases":
            return self._list_cases()
        elif action == "get_case":
            return self._get_case(case_id)
        else:
            return f"Unknown action: {action}. Valid actions are: create_case, update_status, prioritize, list_cases, get_case"
    
    def _create_case(self, transaction_data, assignee=None, risk_factors=None):
        """Create a new investigation case"""
        case_id = f"AML-{self.case_counter}"
        self.case_counter += 1
        
        # Default risk factors if none provided
        if not risk_factors:
            risk_factors = {
                "transaction_amount": 0,
                "customer_risk": "low",
                "geographic_risk": "low",
                "transaction_type_risk": "low"
            }
        
        # Calculate initial priority score
        priority_score = self._calculate_priority(risk_factors)
        
        # Create the case
        case = {
            "case_id": case_id,
            "status": "new",
            "creation_date": datetime.datetime.now().isoformat(),
            "last_updated": datetime.datetime.now().isoformat(),
            "assignee": assignee,
            "transaction_data": transaction_data,
            "risk_factors": risk_factors,
            "priority_score": priority_score,
            "history": [
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "action": "case_created",
                    "details": "New case created"
                }
            ]
        }
        
        self.cases[case_id] = case
        return f"Case {case_id} created with priority score {priority_score}"
    
    def _update_case_status(self, case_id, new_status):
        """Update the status of an existing case"""
        if case_id not in self.cases:
            return f"Case {case_id} not found"
        
        valid_statuses = ["new", "assigned", "in_progress", "pending", "resolved"]
        if new_status not in valid_statuses:
            return f"Invalid status: {new_status}. Valid statuses are: {', '.join(valid_statuses)}"
        
        self.cases[case_id]["status"] = new_status
        self.cases[case_id]["last_updated"] = datetime.datetime.now().isoformat()
        
        # Add to history
        self.cases[case_id]["history"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "action": "status_update",
            "details": f"Status updated to {new_status}"
        })
        
        return f"Case {case_id} status updated to {new_status}"
    
    def _prioritize_case(self, case_id, risk_factors):
        """Recalculate case priority based on risk factors"""
        if case_id not in self.cases:
            return f"Case {case_id} not found"
        
        # Update risk factors if provided
        if risk_factors:
            self.cases[case_id]["risk_factors"] = risk_factors
        
        # Recalculate priority
        priority_score = self._calculate_priority(self.cases[case_id]["risk_factors"])
        self.cases[case_id]["priority_score"] = priority_score
        self.cases[case_id]["last_updated"] = datetime.datetime.now().isoformat()
        
        # Add to history
        self.cases[case_id]["history"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "action": "priority_update",
            "details": f"Priority recalculated to {priority_score}"
        })
        
        return f"Case {case_id} priority updated to {priority_score}"
    
    def _list_cases(self):
        """List all cases sorted by priority"""
        if not self.cases:
            return "No cases found"
        
        # Sort cases by priority (highest first)
        sorted_cases = sorted(
            self.cases.values(), 
            key=lambda x: x["priority_score"], 
            reverse=True
        )
        
        # Format for display
        result = "CASES (sorted by priority):\n"
        for case in sorted_cases:
            result += f"ID: {case['case_id']} | Status: {case['status']} | Priority: {case['priority_score']} | Assignee: {case.get('assignee', 'Unassigned')}\n"
        
        return result
    
    def _get_case(self, case_id):
        """Get detailed information about a specific case"""
        if case_id not in self.cases:
            return f"Case {case_id} not found"
        
        case = self.cases[case_id]
        return json.dumps(case, indent=2)
    
    def _calculate_priority(self, risk_factors):
        """Calculate priority score based on risk factors"""
        # Default score
        score = 0
        
        # Transaction amount factor (higher amount = higher risk)
        amount = risk_factors.get("transaction_amount", 0)
        if amount > 1000000:  # >$1M
            score += 30
        elif amount > 100000:  # >$100K
            score += 20
        elif amount > 10000:  # >$10K
            score += 10
        
        # Customer risk factor
        customer_risk = risk_factors.get("customer_risk", "low").lower()
        if customer_risk == "high":
            score += 30
        elif customer_risk == "medium":
            score += 15
        
        # Geographic risk factor
        geo_risk = risk_factors.get("geographic_risk", "low").lower()
        if geo_risk == "high":
            score += 20
        elif geo_risk == "medium":
            score += 10
        
        # Transaction type risk
        tx_type_risk = risk_factors.get("transaction_type_risk", "low").lower()
        if tx_type_risk == "high":
            score += 20
        elif tx_type_risk == "medium":
            score += 10
        
        return score


class ComplianceAnalystToolInput(BaseModel):
    """Input schema for Compliance Analyst Tool."""
    transaction_data: dict = Field(..., description="Transaction data to be checked")
    check_type: str = Field(..., description="Type of check to perform: 'sanctions', 'pep', or 'both'")
    entity_name: str = Field(None, description="Name of individual or entity to check (optional)")
    entity_country: str = Field(None, description="Country of the individual or entity (optional)")

class ComplianceAnalystTool(BaseTool):
    name = "Compliance Analyst Tool"
    description = "This tool validates transactions against global sanction lists and PEP (Politically Exposed Persons) databases."
    args_schema: Type[BaseModel] = ComplianceAnalystToolInput
    
    def __init__(self):
        super().__init__()
        # Mock databases - would be replaced with real APIs in production
        self.sanctions_db = self._initialize_sanctions_db()
        self.pep_db = self._initialize_pep_db()
    
    def _run(self, transaction_data: dict, check_type: str, entity_name: str = None, entity_country: str = None) -> str:
        results = {}
        
        if check_type.lower() in ['sanctions', 'both']:
            sanctions_results = self._check_sanctions(transaction_data, entity_name, entity_country)
            results['sanctions_check'] = sanctions_results
        
        if check_type.lower() in ['pep', 'both']:
            pep_results = self._check_pep(transaction_data, entity_name, entity_country)
            results['pep_check'] = pep_results
        
        # Determine overall compliance status
        if 'sanctions_check' in results and results['sanctions_check']['match_found']:
            results['compliance_status'] = 'BLOCKED - SANCTIONS MATCH'
            results['risk_level'] = 'high'
        elif 'pep_check' in results and results['pep_check']['match_found']:
            results['compliance_status'] = 'FLAGGED - PEP MATCH'
            results['risk_level'] = 'medium'
        else:
            results['compliance_status'] = 'CLEARED'
            results['risk_level'] = 'low'
        
        # Add recommendation based on findings
        results['recommendation'] = self._generate_recommendation(results)
        
        return str(results)
    
    def _check_sanctions(self, transaction_data, entity_name=None, entity_country=None):
        """Check transaction against sanctions lists"""
        # Extract relevant data for checking
        sender_account = transaction_data.get('Account', [None])[0] if isinstance(transaction_data.get('Account', None), list) else transaction_data.get('Account', None)
        receiver_account = transaction_data.get('Account.1', [None])[0] if isinstance(transaction_data.get('Account.1', None), list) else transaction_data.get('Account.1', None)
        
        # Name takes precedence if provided
        if entity_name:
            # Check if entity name is in sanctions list
            match = any(entity_name.lower() in sanctioned_entity['name'].lower() for sanctioned_entity in self.sanctions_db)
            if match:
                matched_entities = [entity for entity in self.sanctions_db if entity_name.lower() in entity['name'].lower()]
                return {
                    'match_found': True,
                    'matched_entities': matched_entities,
                    'match_type': 'name',
                    'confidence': 'high' if any(entity_name.lower() == entity['name'].lower() for entity in matched_entities) else 'medium'
                }
        
        # Check accounts against sanctions list
        for account in [sender_account, receiver_account]:
            if not account:
                continue
                
            for entity in self.sanctions_db:
                if 'accounts' in entity and account in entity['accounts']:
                    return {
                        'match_found': True,
                        'matched_entities': [entity],
                        'match_type': 'account',
                        'confidence': 'high'
                    }
        
        # No matches found
        return {
            'match_found': False,
            'matched_entities': [],
            'match_type': None,
            'confidence': None
        }
    
    def _check_pep(self, transaction_data, entity_name=None, entity_country=None):
        """Check transaction against PEP database"""
        # Similar to sanctions check but for PEP
        sender_account = transaction_data.get('Account', [None])[0] if isinstance(transaction_data.get('Account', None), list) else transaction_data.get('Account', None)
        receiver_account = transaction_data.get('Account.1', [None])[0] if isinstance(transaction_data.get('Account.1', None), list) else transaction_data.get('Account.1', None)
        
        # Name takes precedence if provided
        if entity_name:
            # Check if entity name is in PEP list
            match = any(entity_name.lower() in pep_entity['name'].lower() for pep_entity in self.pep_db)
            if match:
                matched_entities = [entity for entity in self.pep_db if entity_name.lower() in entity['name'].lower()]
                return {
                    'match_found': True,
                    'matched_entities': matched_entities,
                    'match_type': 'name',
                    'confidence': 'high' if any(entity_name.lower() == entity['name'].lower() for entity in matched_entities) else 'medium'
                }
        
        # Check accounts against PEP list
        for account in [sender_account, receiver_account]:
            if not account:
                continue
                
            for entity in self.pep_db:
                if 'accounts' in entity and account in entity['accounts']:
                    return {
                        'match_found': True,
                        'matched_entities': [entity],
                        'match_type': 'account',
                        'confidence': 'high'
                    }
        
        # No matches found
        return {
            'match_found': False,
            'matched_entities': [],
            'match_type': None,
            'confidence': None
        }
    
    def _generate_recommendation(self, results):
        """Generate recommendation based on check results"""
        if results.get('compliance_status') == 'BLOCKED - SANCTIONS MATCH':
            return "Transaction must be blocked. File a Suspicious Activity Report (SAR) and notify the compliance officer immediately."
        elif results.get('compliance_status') == 'FLAGGED - PEP MATCH':
            return "Enhanced due diligence required. Collect additional documentation on source of funds and purpose of transaction before proceeding."
        else:
            return "Transaction cleared for compliance. No further action needed from a regulatory perspective."
    
    def _initialize_sanctions_db(self):
        """Initialize mock sanctions database"""
        return [
            {
                'name': 'Restricted Trading Company',
                'country': 'Country A',
                'list': 'OFAC SDN',
                'reason': 'Proliferation financing',
                'accounts': ['783920456', '100428999']
            },
            {
                'name': 'Global Sanctioned Bank',
                'country': 'Country B',
                'list': 'EU Consolidated',
                'reason': 'Supporting terrorism',
                'accounts': ['892375610']
            },
            {
                'name': 'John Restricted Doe',
                'country': 'Country C',
                'list': 'UN Sanctions',
                'reason': 'Human rights violations',
                'accounts': ['567123890']
            },
            {
                'name': 'International Blocked Corp',
                'country': 'Country D',
                'list': 'UK Sanctions',
                'reason': 'Sanctions evasion',
                'accounts': ['456789012']
            }
        ]
    
    def _initialize_pep_db(self):
        """Initialize mock PEP database"""
        return [
            {
                'name': 'Minister Finance',
                'country': 'Country E',
                'position': 'Finance Minister',
                'risk_level': 'high',
                'accounts': ['100428660']  # Matching our test account
            },
            {
                'name': 'Governor Central',
                'country': 'Country F',
                'position': 'Central Bank Governor',
                'risk_level': 'medium',
                'accounts': ['234567891']
            },
            {
                'name': 'Diplomat General',
                'country': 'Country G',
                'position': 'Ambassador',
                'risk_level': 'low',
                'accounts': ['345678912']
            },
            {
                'name': 'Party Leader',
                'country': 'Country H',
                'position': 'Political Party Leader',
                'risk_level': 'medium',
                'accounts': ['456789123']
            }
        ]

