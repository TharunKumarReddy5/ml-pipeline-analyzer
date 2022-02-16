from diagrams.aws.cost import CostAndUsageReport
from diagrams.aws.network import ELB
from diagrams.gcp.compute import Functions


class PipelineNode:
    def __init__(self, node_type, title, desc):
        self.type = node_type
        self.title = title
        self.desc = desc
        self.view = True

    @staticmethod
    def __create_node(node_names):
        if node_names == 'custom function':
            return Functions(node_names)
        elif node_names == 'svm':
            return ELB(node_names)
        elif node_names == 'impute':
            return CostAndUsageReport(node_names)
        else:
            return Functions(node_names)
