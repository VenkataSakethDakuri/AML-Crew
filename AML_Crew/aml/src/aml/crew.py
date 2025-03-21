from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from aml.tools.custom_tool import PredictorTool, ReportTool



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

    @task
    def Prediction_task(self) -> Task:
        return Task(
            config=self.tasks_config['Prediction_task'],
        )

    @task
    def Reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            #output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Aml crew"""


        return Crew(
            agents=self.agents,
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
