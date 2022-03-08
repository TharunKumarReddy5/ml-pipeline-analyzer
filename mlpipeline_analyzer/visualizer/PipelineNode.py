from diagrams.aws.cost import CostAndUsageReport
from diagrams.aws.network import ELB
from diagrams.gcp.compute import Functions, AppEngine
from diagrams.gcp.ml import RecommendationsAI, NaturalLanguageAPI, JobsAPI, InferenceAPI
from diagrams.aws.database import Database
from diagrams.gcp.analytics import Dataflow
import regex as re


class PipelineNode:
    """
    Pipeline Node class that creates a node for each component in the pipeline and maps it to the diagrams object.
    """
    def __init__(self):
        self.node = None
        self.name = None

    def create_node(self, node_name):
        """
        Creates the nodes for the summary diagram.

        Args:
            node_name (list): List of base component name and the pipeline component name.

        Returns:
            obj: Diagram node object.
        """
        self.node = node_name[0]
        self.name = re.sub(r'\(.*\)', '', str(node_name[1]))

        if self.node == 'Custom Function':
            return Functions(self.node)
        elif self.node in ('data', 'datasets'):
            return Database(self.name)
        elif self.node == 'Data Stream':
            return Dataflow("Data Stream")
        elif self.node in ('compose', 'covariance', 'preprocessing', 'kernel_approximation', 'transformers'):
            return NaturalLanguageAPI(self.name)
        elif self.node in ('feature_selection', 'feature_extraction', 'manifold', 'random_projection'):
            return RecommendationsAI(self.name)
        elif self.node in ('cross_decomposition', 'decomposition', 'discriminant_analysis'):
            return AppEngine(self.name)
        elif self.node in ('cluster', 'ensemble', 'gaussian_process', 'isotonic', 'kernel_ridge', 'linear_model',
                           'mixture', 'multiclass', 'multioutput', 'naive_bayes', 'neighbors', 'neural_network',
                           'semi_supervised', 'svm', 'tree', 'gradient_boosting', 'xgboost', 'estimators'):
            return ELB(self.name)
        elif self.node in 'model_selection':
            return InferenceAPI(self.name)
        elif self.node in 'impute':
            return CostAndUsageReport(self.name)
        elif self.node in ('inspection', 'metrics'):
            return JobsAPI(self.name)
        else:
            return Functions(self.node)
