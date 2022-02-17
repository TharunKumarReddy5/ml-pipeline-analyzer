from diagrams import Cluster, Diagram
from .PipelineNode import PipelineNode
import sklearn
from sklearn import *
import regex as re
import warnings
#warnings.filterwarnings("ignore")

class PipelineDiagram:
    def __init__(self, pipeline):
        self.pipe = pipeline
        self.title = 'Machine Learning Pipeline'
        self.view = True
        self.cn = PipelineNode()

    def show(self, title=None):
        self.title = title if title else self.title
        self.pipe_len = len(list(self.pipe))
        return self.create_diagram()

    @staticmethod
    def parent_classes(level=0, base='sklearn'):
        if level != 0:
            base = 'sklearn.' + base
        return list(filter(lambda x: not re.search(r'^_.*', x), dir(eval(base))))

    def all_classes(self):
        l = self.parent_classes()
        for i in self.parent_classes():
            try:
                eval(i)
            except:
                l.remove(i)
        class_list = {x: [eval('sklearn.' + x + '.' + y) for y in self.parent_classes(1, x)] for x in l}
        return class_list

    def get_link(self, path):
        reg = re.findall(r"'(.*)'", str(path))[0]
        link = 'https://scikit-learn.org/stable/modules/generated/{0}.html'.format(re.sub("".join(re.findall(r'\.(_.*\.)',reg)),'',reg))
        return link
    
    def find_category(self, obj):
        temp = self.all_classes()
        try:
            comp = str(type(obj)).split('.')[1]
            if type(obj) in temp[comp] and comp!='pipeline':
                return (comp.title(), obj, self.get_link(type(obj)))
            if comp=='pipeline':
                return list(map(self.find_category, [x[1] for x in obj.transformer_list]))
        except:
            return ('Custom Function', obj, 'Function')
            
    def all_categories(self):
        return list(map(self.find_category, self.pipe))

    def create_diagram(self):
        with Diagram(self.title, show=False) as pipe_diag:
            inputs = [("Train","Train Data"), ("Validation", "Validation Data"), ("Test","Test Data")]
            start = self.create_cluster("Input Data", inputs) >> self.cn.create_node(("Data Stream","Data Stream"))
            self.traverse_pipeline(start)
        return pipe_diag

    def traverse_pipeline(self, curr):
        self.descriptions = list(self.all_categories())
        for i in self.descriptions:
            if type(i) == list:
                curr = curr >> self.create_cluster("Transformers", i)
            else:
                curr = curr >> self.cn.create_node(i)
        return curr

    def create_cluster(self, cluster_name, node_names):
        with Cluster(cluster_name):
            return list(map(self.cn.create_node, node_names))
        
    def show_params(self):
        return self.title
