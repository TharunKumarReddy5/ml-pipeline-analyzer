from diagrams.aws.cost import CostAndUsageReport
from diagrams.aws.network import ELB
from diagrams.gcp.compute import Functions, AppEngine
from diagrams.gcp.ml import RecommendationsAI, NaturalLanguageAPI, JobsAPI, InferenceAPI
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
        elif self.node in ('data','datasets'):
            return Database(self.name)
        elif self.node =='Data Stream':
            return Dataflow("Data Stream")
        elif self.node in ('compose','covariance','preprocessing','kernel_approximation'):
            return NaturalLanguageAPI(self.name)
        elif self.node in ('feature_selection', 'feature_extraction', 'manifold', 'random_projection'):
            return RecommendationsAI(self.name)
        elif self.node in ('cross_decomposition','decomposition','discriminant_analysis'):
            return AppEngine(self.name)
        elif self.node in ('cluster', 'ensemble', 'gaussian_process', 'isotonic', 'kernel_ridge', 'linear_model', 'mixture', 'multiclass', 'multioutput', 
                           'naive_bayes','neighbors', 'neural_network', 'semi_supervised', 'svm', 'tree', 'gradient_boosting', 'xgboost'):
            return ELB(self.name)
        elif self.node in ('model_selection'):
            return InferenceAPI(self.name)
        elif self.node in ('Impute'):
            return CostAndUsageReport(self.name)
        elif self.node in ('inspection', 'metrics'):
            return JobsAPI(self.name)
        else:
            return Functions(self.node)