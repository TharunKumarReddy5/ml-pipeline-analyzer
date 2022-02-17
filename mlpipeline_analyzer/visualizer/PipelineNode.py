from diagrams.aws.cost import CostAndUsageReport
from diagrams.aws.network import ELB
from diagrams.gcp.compute import Functions, AppEngine
from diagrams.gcp.ml import RecommendationsAI, NaturalLanguageAPI
from diagrams.aws.database import Database
from diagrams.gcp.analytics import BigQuery, Dataflow, PubSub
from diagrams.gcp.database import BigTable
from diagrams.gcp.iot import IotCore
from diagrams.gcp.storage import GCS
import regex as re

class PipelineNode:
    '''
    def __init__(self):
        return self
'''
    def create_node(self, node_name):
        self.node = node_name[0]
        self.name = re.sub(r'\(.*\)', '',  str(node_name[1]))
                
        if self.node == 'Custom Function':
            return Functions(self.node)
        elif self.node in ('Train', 'Test', 'Validation'):
            return Database(self.name)
        elif self.node =='Data Stream':
            return Dataflow("Data Stream")
        elif self.node == 'Preprocessing':
            return NaturalLanguageAPI(self.name)
        elif self.node == 'Feature_Selection':
            return RecommendationsAI(self.name)
        elif self.node in ('Discriminant_Analysis','Decomposition'):
            return AppEngine(self.name)
        elif self.node == 'Svm':
            return ELB(self.name)
        elif self.node in ('Tree', 'Ensemble'):
            return ELB(self.name)
        elif self.node == 'Impute':
            return CostAndUsageReport(self.name)
        else:
            return Functions(self.node)