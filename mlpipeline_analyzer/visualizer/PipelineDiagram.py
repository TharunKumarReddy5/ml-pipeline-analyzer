from diagrams import Cluster, Diagram
from graphviz import Digraph
from .PipelineNode import PipelineNode
import sklearn
from sklearn import *
import regex as re
import warnings
#warnings.filterwarnings("ignore")


class PipelineDiagram:
    def __init__(self, pipeline, file_name='ml_pipeline.png'):
        self.pipe = pipeline
        self.title = 'Machine Learning Pipeline'
        self.title_param = 'Machine Learning Parameters Pipeline'
        self.view = True
        self.file_name = file_name
        self.cn = PipelineNode()

    def show(self, title=None):
        self.title = title if title else self.title
        self.pipe_len = len(list(self.pipe))
        return self.create_diagram()
    
    def show_params(self, title=None):
        self.title_param = title if title else self.title_param
        return self.create_param_diagram()

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
                return (comp, obj, self.get_link(type(obj)))
            if comp=='pipeline':
                return list(map(self.find_category, [x[1] for x in obj.transformer_list]))
        except:
            return ('Custom Function', obj, 'Function')
    
    def find_category_params(self, obj):
        try:
            comp = str(type(obj)).split('.')[1]
            if comp!='pipeline':
                return (obj, self.get_param(obj))
            if comp=='pipeline':
                return list(map(self.find_category_params, [x[1] for x in obj.transformer_list]))
        except:
            return (obj, 'Custom Function')
    
    def get_param(self, obj):
        try:
            s = list(obj.get_params().items())
            reg = re.sub(r"(,\s)\'","\l'",str(dict(filter(lambda x: '__' not in x[0] , s))))
            return re.sub('(\(.*\))', '', str(obj))+'\n\n'+re.sub('{|}', '', reg)
        except:
            return str(obj)
    
    def all_params(self):
        return list(map(self.find_category_params, self.pipe))
            
    def all_categories(self):
        return list(map(self.find_category, self.pipe))

    def create_diagram(self):
        with Diagram(self.title, show=False, filename=self.file_name) as pipe_diag:
            inputs = [("data","Train Data"), ("data", "Validation Data"), ("data","Test Data")]
            start = self.create_cluster("Input Data", inputs) >> self.cn.create_node(("Data Stream","Data Stream"))
            self.traverse_pipeline(start)
        return pipe_diag
    
    def create_param_diagram(self):
        self.g = Digraph('G', filename='ml_pipeline_params.gv')
        self.g.graph_attr["rankdir"] = "LR"
        self.create_cluster_params('Inputs', ['Train Data', 'Validation Data', 'Test Data'])      
        #self.g.edge('input','streamin')
        #self.g.edge('streamout','Model')
        self.traverse_pipeline_params()
        self.g.view()
        return self

    def traverse_pipeline(self, curr):
        self.descriptions = list(self.all_categories())
        for i in self.descriptions:
            if type(i) == list:
                curr = curr >> self.create_cluster("Transformers", i)
            else:
                curr = curr >> self.cn.create_node(i)
        return curr

    def traverse_pipeline_params(self):
        self.params = self.all_params()
        for i in self.params:
            if type(i) == list:
                self.create_cluster_params('Transformers', [x[1] for x in i])
            else:
                self.g.node(str(i[0]), label=i[1], shape='box')
                self.g.edge(self.input, str(i[0]))
                self.input = str(i[0])
        return self
        
    def create_cluster(self, cluster_name, node_names):
        with Cluster(cluster_name):
            return list(map(self.cn.create_node, node_names))
        
    def create_cluster_params(self, cluster_name, node_names):
        with self.g.subgraph(name='cluster_'+cluster_name) as c:
            inlabel = 'streamin_' + cluster_name
            outlabel = 'streamout_' + cluster_name
            c.attr(style='filled', color='green', URL='https://stackoverflow.com')
            c.node_attr.update(style='filled', color='white')
            c.node(outlabel, label='Stream', shape='box')
            if cluster_name != 'Inputs':
                c.node(inlabel, label='Stream', shape='box')
                self.g.edge(self.input, inlabel)
                c.node(outlabel, label='Union', shape='box')
            for i in range(len(node_names)):
                c.node(cluster_name+str(i), label=node_names[i], shape='box')
                if cluster_name!='Inputs':
                    c.edge(inlabel, str(cluster_name+str(i)))
                c.edge(cluster_name+str(i), outlabel)
            self.input = outlabel
            c.attr(label=cluster_name, URL='https://stackoverflow.com')
        
        
