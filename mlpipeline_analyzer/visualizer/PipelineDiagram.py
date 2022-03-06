from diagrams import Cluster, Diagram
from graphviz import Digraph
from .PipelineNode import PipelineNode
import sklearn
import evalml
from sklearn import *
from evalml.pipelines.components import *
import regex as re
import warnings
#warnings.filterwarnings("ignore")


class PipelineDiagram:
    def __init__(self, pipeline, file_name='ml_pipeline'):
        self.pipe = pipeline
        self.view = True
        self.file_name = file_name
        self.cn = PipelineNode()
        if str(type(self.pipe)).split(' ')[1].split('.')[0].replace("'","") == 'evalml':
            self.base = 'evalml.pipelines.components'
        else:
            self.base = 'sklearn'

    def show(self, title='Machine Learning Pipeline'):
        self.title = title
        self.pipe_len = len(list(self.pipe))
        return self.create_diagram()
    
    def show_params(self, title='Machine Learning Parameters Pipeline'):
        self.title_param = title
        return self.create_param_diagram()

    def parent_classes(self, base=None):
        base = base if base!=None else self.base
        return list(filter(lambda x: not re.search(r'^_.*', x), dir(eval(base))))

    def all_classes(self):
        l = self.parent_classes()
        for i in self.parent_classes():
            try:
                eval(i)
            except:
                l.remove(i)
        class_list = {x: [eval(self.base+'.' + x + '.' + y) for y in self.parent_classes(self.base+'.'+x) if y!='default_parameters'] for x in l}
        return class_list

    def get_link(self, path):
        reg = re.findall(r"'(.*)'", str(path))[0]
        if self.base=='evalml.pipelines.components':
            root_url = 'https://evalml.alteryx.com/en/stable/autoapi/{0}/index.html#{1}'
            r1 = re.sub("".join(re.findall(r'\.(\w+_.*)',reg)),'',reg)
            r1 = "/".join(r1.split('.')[:-1])
            r2 = re.sub("".join(re.findall(r'\.(\w+_.*\.)',reg)),'',reg)
            link = root_url.format(r1,r2)
        else:
            root_url = 'https://scikit-learn.org/stable/modules/generated/{0}.html'
            link = root_url.format(re.sub("".join(re.findall(r'\.(_.*\.)',reg)),'',reg))
        return link
    
    def find_category(self, obj):
        temp = self.all_classes()
        try:
            comp = str(type(obj)).split('.')[3] if self.base=='evalml.pipelines.components' else str(type(obj)).split('.')[1]
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
                return (obj, (self.get_param(obj), self.get_link(type(obj))))
            if comp=='pipeline':
                return list(map(self.find_category_params, [x[1] for x in obj.transformer_list]))
        except:
            return (obj, ('Custom Function',''))
    
    def get_param(self, obj):
        try:
            s = list(obj.parameters.items() if self.base=='evalml.pipelines.components' else obj.get_params().items())
            reg = re.sub(r"(,\s)\'","\l'",str(dict(filter(lambda x: '__' not in x[0] , s))))
            return re.sub('(\(.*\))', '', str(obj))+' |'+re.sub('{|}', '', reg)
        except:
            return str(obj)
    
    def all_categories(self):
        return list(map(self.find_category, self.pipe))
    
    def all_params(self):
        return list(map(self.find_category_params, self.pipe))

    def create_diagram(self):
        with Diagram(self.title, show=False, filename=self.file_name) as pipe_diag:
            inputs = [("data","Train Data"), ("data", "Validation Data"), ("data","Test Data")]
            start = self.create_cluster("Input Data", inputs) >> self.cn.create_node(("Data Stream","Data Stream"))
            self.traverse_pipeline(start)
        return pipe_diag
    
    def create_param_diagram(self):
        self.g = Digraph(name=self.title_param, filename='ml_pipeline_params', graph_attr={"splines": "true", "overlap": "scale", "rankdir": "LR"})
        self.create_cluster_params('Inputs', [('Train Data',''), ('Validation Data',''), ('Test Data','')])
        self.traverse_pipeline_params()
        #self.g.view()
        return self.g

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
                self.g.node(str(i[0]), label=i[1][0], URL=i[1][1], shape='record', nodesep='0.03')
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
            c.attr(label=cluster_name, style='filled', color='cadetblue2', shape='record', nodesep="0.03")
            c.node(outlabel, label='Stream', shape='box', color='white')
            if cluster_name != 'Inputs':
                c.node(inlabel, label='Stream', shape='box', color='white')
                self.g.edge(self.input, inlabel)
                c.node(outlabel, label='Union', shape='box', color='white')
            for i in range(len(node_names)):
                c.node(cluster_name+str(i), label=node_names[i][0], URL=node_names[i][1])
                if cluster_name!='Inputs':
                    c.edge(inlabel, str(cluster_name+str(i)))
                c.edge(cluster_name+str(i), outlabel)
            c.node_attr.update(style='filled', color='white', shape='record', nodesep='0.03')
            self.input = outlabel
        
        
