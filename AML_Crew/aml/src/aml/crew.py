from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from aml.tools.custom_tool import PredictorTool, ReportTool, CaseManagerTool, ComplianceAnalystTool



@CrewBase
class Aml():
    """Aml crew"""


    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

#test

    

    @agent
    def Predictor(self) -> Agent:
        return Agent(
            config=self.agents_config['Predictor'],
            tools = [PredictorTool()],
            verbose=True
        )
    



    

    @agent
    def Analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['Analyst'],
            tools = [ReportTool()],
            verbose=True
        )

    @agent
    def CaseManager(self) -> Agent:
        return Agent(
            config=self.agents_config['CaseManager'],
            tools = [CaseManagerTool()],
            verbose=True
        )

    @agent
    def ComplianceAnalyst(self) -> Agent:
        return Agent(
            config=self.agents_config['ComplianceAnalyst'],
            tools = [ComplianceAnalystTool()],
            verbose=True
        )
    

    @task
    def Sanctions_check_task(self) -> Task:
        return Task(
            config=self.tasks_config['Sanctions_check_task'],
        )
    
    @task
    def PEP_check_task(self) -> Task:
        return Task(
            config=self.tasks_config['PEP_check_task'],
        )
    
    @task
    def Compliance_validation_task(self) -> Task:
        return Task(
            config=self.tasks_config['Compliance_validation_task'],
        )




    @task
    def Prediction_task(self) -> Task:
        return Task(
            config=self.tasks_config['Prediction_task'],
        )

    @task
    def Reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['Reporting_task'],
        )

    @task
    def Case_creation_task(self) -> Task:
        return Task(
            config=self.tasks_config['Case_creation_task'],
        )
    
    @task
    def Case_update_task(self) -> Task:
        return Task(
            config=self.tasks_config['Case_update_task'],
        )
    
    @task
    def Case_prioritization_task(self) -> Task:
        return Task(
            config=self.tasks_config['Case_prioritization_task'],
        )
    
    @task
    def Case_management_task(self) -> Task:
        return Task(
            config=self.tasks_config['Case_management_task'],
        )


    @crew
    def crew(self) -> Crew:
        """Creates the Aml crew"""


        return Crew(
            agents=self.agents,
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical,https://docs.crewai.com/how-to/Hierarchical/
        )
