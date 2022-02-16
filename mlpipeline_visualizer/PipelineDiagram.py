from diagrams import Cluster, Diagram
from diagrams.gcp.analytics import BigQuery, Dataflow, PubSub
from diagrams.gcp.compute import AppEngine
from diagrams.aws.database import Database
from diagrams.aws.cost import CostAndUsageReport
from diagrams.gcp.database import BigTable
from diagrams.gcp.iot import IotCore
from diagrams.gcp.storage import GCS
import sklearn
from sklearn import *
import regex as re


class PipelineDiagram:
    def __init__(self, pipeline):
        self.pipe = pipeline
        self.title = "ML Pipeline"
        self.view = True

    def show(self, title=None):
        self.title = title if title else self.title
        self.pipe_len = len(list(self.pipe))
        return self.__create_diagram()

    @staticmethod
    def __parent_classes(level=0, base='sklearn'):
        if level != 0:
            base = 'sklearn.' + base
        return list(filter(lambda x: not re.search(r'^_.*', x), dir(eval(base))))

    def __all_classes(self):
        l = self.__parent_classes()
        for i in self.__parent_classes():
            try:
                eval(i)
            except:
                l.remove(i)
        class_list = {x: [eval('sklearn.' + x + '.' + y) for y in self.__parent_classes(1, x)] for x in l}
        return class_list

    def __find_category(self, obj):
        temp = self.__all_classes()
        for i in temp:
            if type(obj) in temp[i]:
                if i != 'pipeline':
                    return i
                else:
                    return list(map(self.__find_category, [x[1] for x in obj.transformer_list]))
        return 'custom function'

    def __all_categories(self):
        return list(map(self.__find_category, self.pipe))

    def __create_diagram(self):
        with Diagram(self.title, show=False) as pipe_diag:
            start = self.__data_collection() >> Dataflow("Data Stream")
            self.__traverse_pipeline(start)
        return pipe_diag

    def __traverse_pipeline(self, curr):
        self.__descriptions = list(self.__all_categories())
        for i in self.__descriptions:
            if type(i) == list:
                curr = curr >> self.__create_cluster(node_names=i)
            else:
                curr = curr >> self.__create_node(node_names=i)
        return curr

    @staticmethod
    def __create_cluster(node_names):
        with Cluster("Transformers"):
            return [AppEngine(i) for i in node_names]

    @staticmethod
    def __data_collection():
        with Cluster("Input Data"):
            return [Database("Train Data"), Database("Validation Data"), Database("Test Data")]
