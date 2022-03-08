import sklearn
from sklearn import *
import evalml
from evalml.pipelines.components import *
from diagrams import Cluster, Diagram
from graphviz import Digraph
from .PipelineNode import PipelineNode
import regex as re


class PipelineDiagram:
    """
    Pipeline Diagram class that creates a summary diagram of the pipeline and hyperparameter flow diagram of the pipeline.
    """
    def __init__(self, pipeline, file_name='ml_pipeline'):
        self.pipe = pipeline
        self.view = True
        self.file_name = file_name
        self.cn = PipelineNode()
        if str(type(self.pipe)).split(' ')[1].split('.')[0].replace("'", "") == 'evalml':
            self.base = 'evalml.pipelines.components'
        else:
            self.base = 'sklearn'

    def show(self, title='Machine Learning Pipeline'):
        """
        Generates summary diagram of the input sklearn or evalml pipeline.

        Args:
            title (str, optional): Title for the summary pipeline diagram. Defaults to 'Machine Learning Parameters Pipeline'.. Defaults to 'Machine Learning Pipeline'.

        Returns:
            image: Image with summary diagram connecting each component in the pipeline.
        """
        self.title = title
        self.pipe_len = len(list(self.pipe))
        return self._create_diagram()

    def show_params(self, title='Machine Learning Parameters Pipeline'):
        """
        Generates diagram of the pipeline with hyperparameters with embedded links to the documentation in sklearn and evalml.

        Args:
            title (str, optional): Title for the pipeline parameters flow diagram. Defaults to 'Machine Learning Parameters Pipeline'.

        Returns:
            image: Image with hyperparameter diagram with embedded hyperlinks for each component in the pipeline.
        """
        self.title_param = title
        return self._create_param_diagram()

    def _parent_classes(self, base=None):
        """
        Generates a list of all the avaiable parent classes of the input pipeline object.

        Args:
            base (obj, optional): Takes values of either sklearn or evalml.pipelines.components. Defaults to None.

        Returns:
            list: List of all the parent classes of the input pipeline object.
        """
        base = base if base is not None else self.base
        return list(filter(lambda x: not re.search(r'^_.*', x), dir(eval(base))))

    def _all_classes(self):
        """
        Generates a list of all the avaiable base classes for each parent class of the input pipeline object.

        Returns:
            list: List of all the base classes of the respective parent class.
        """
        parent_list = self._parent_classes()
        for i in self._parent_classes():
            try:
                eval(i)
            except NameError:
                parent_list.remove(i)
        class_list = {x: [eval(self.base + '.' + x + '.' + y) for y in self._parent_classes(self.base + '.' + x) if
                          y != 'default_parameters'] for x in parent_list}
        return class_list

    def _get_link(self, path):
        """
        Generates the documentation API reference link for the input sklearn or evalml object.

        Args:
            path (obj): Takes the pipeline component object path from either sklearn or evalml.pipelines.components.

        Returns:
            str: Link to the documentation API reference for the input pipeline component.
        """
        reg = re.findall(r"'(.*)'", str(path))[0]
        if self.base == 'evalml.pipelines.components':
            root_url = 'https://evalml.alteryx.com/en/stable/autoapi/{0}/index.html#{1}'
            r1 = re.sub("".join(re.findall(r'\.(\w+_.*)', reg)), '', reg)
            r1 = "/".join(r1.split('.')[:-1])
            r2 = re.sub("".join(re.findall(r'\.(\w+_.*\.)', reg)), '', reg)
            link = root_url.format(r1, r2)
        else:
            root_url = 'https://scikit-learn.org/stable/modules/generated/{0}.html'
            link = root_url.format(re.sub("".join(re.findall(r'\.(_.*\.)', reg)), '', reg))
        return link

    def _find_category(self, obj):
        """
        Finds the base class category of the input sklearn or evalml object and extracts the URL for the pipeline component.

        Args:
            obj (obj): Takes the pipeline component object from either sklearn or evalml pipeline.

        Returns:
            tuple: tuple with the base class name, category of the input pipeline component and the URL for the pipeline component.
        """
        temp = self._all_classes()
        try:
            comp = str(type(obj)).split('.')[3] if self.base == 'evalml.pipelines.components' else \
                str(type(obj)).split('.')[1]
            if type(obj) in temp[comp] and comp != 'pipeline':
                return comp, obj, self._get_link(type(obj))
            if comp == 'pipeline':
                return list(map(self._find_category, [x[1] for x in obj.transformer_list]))
        except IndexError:
            return 'Custom Function', obj, 'Function'

    def _find_category_params(self, obj):
        """
        Finds the category of the input sklearn or evalml object and extracts the hyperparameters and URL for the pipeline component.

        Args:
            obj (obj): Takes the pipeline component object from either sklearn or evalml pipeline.

        Returns:
            tuple: tuple with the category of the input pipeline component and the hyperparameters and URL for the pipeline component.
        """
        try:
            comp = str(type(obj)).split('.')[1]
            if comp != 'pipeline':
                return obj, (self._get_param(obj), self._get_link(type(obj)))
            if comp == 'pipeline':
                return list(map(self._find_category_params, [x[1] for x in obj.transformer_list]))
        except IndexError:
            return obj, ('Custom Function', '')

    def _get_param(self, obj):
        """
        Creates a dictionary string of the hyperparameters of the input sklearn or evalml object.

        Args:
            obj (obj): Takes the pipeline component object from either sklearn or evalml pipeline.

        Returns:
            str: Dictionary string of the hyperparameters of the input pipeline component.
        """
        try:
            s = list(obj.parameters.items() if self.base == 'evalml.pipelines.components' else obj._get_params().items())
            reg = re.sub(r"(,\s)\'", r"\l'", str(dict(filter(lambda x: '__' not in x[0], s))))
            return re.sub(r'(\(.*\))', '', str(obj)) + ' |' + re.sub('{|}', '', reg)
        except (NameError, IndexError, ReferenceError, Exception):
            return str(obj)

    def _all_categories(self):
        """
        Generates a list of all the categories for the components in the pipeline.

        Returns:
            list: list of all categories for the components in the pipeline.
        """
        return list(map(self._find_category, self.pipe))

    def _all_params(self):
        """
        Generates a list of all the avaiable hyperparameters for the components in the pipeline.

        Returns:
            list: list of all hyperparameters for the components in the pipeline.
        """
        return list(map(self._find_category_params, self.pipe))

    def _create_diagram(self):
        """
        Main function that initiates the creation of summary diagram of the pipeline.

        Returns:
            image: Image with summary diagram connecting each component in the pipeline.
        """
        with Diagram(self.title, show=False, filename=self.file_name) as pipe_diag:
            inputs = [("data", "Train Data"), ("data", "Validation Data"), ("data", "Test Data")]
            start = self._create_cluster("Input Data", inputs) >> self.cn.create_node(("Data Stream", "Data Stream"))
            self._traverse_pipeline(start)
        return pipe_diag

    def _create_param_diagram(self):
        """
        Main function that initiates the creation of hyperparameter diagram of the pipeline.

        Returns:
            image: Image with hyperparameter diagram with embedded hyperlinks for each component in the pipeline.
        """
        self.g = Digraph(name=self.title_param, filename='ml_pipeline_params',
                         graph_attr={"splines": "true", "overlap": "scale", "rankdir": "LR"})
        self._create_cluster_params('Inputs', [('Train Data', ''), ('Validation Data', ''), ('Test Data', '')])
        self._traverse_pipeline_params()
        # self.g.view()
        return self.g

    def _traverse_pipeline(self, curr):
        """
        Traverses the pipeline and creates a graph of the pipeline for summary diagram.
        """
        self.descriptions = list(self._all_categories())
        for i in self.descriptions:
            if type(i) == list:
                curr = curr >> self._create_cluster("Transformers", i)
            else:
                curr = curr >> self.cn.create_node(i)
        return curr

    def _traverse_pipeline_params(self):
        """
        Traverses the pipeline and creates a graph of the pipeline for hyperparameter diagram.
        """
        self.params = self._all_params()
        for i in self.params:
            if type(i) == list:
                self._create_cluster_params('Transformers', [x[1] for x in i])
            else:
                self.g.node(str(i[0]), label=i[1][0], URL=i[1][1], shape='record', nodesep='0.03')
                self.g.edge(self.input, str(i[0]))
                self.input = str(i[0])
        return self

    def _create_cluster(self, cluster_name, node_names):
        """
        Creates a cluster for summary diagram with the given name and groups the given nodes to it.

        Args:
            cluster_name (str): Name of the cluster to be formed.
            node_names (list): two dimensional list containing the elements like Node Name, URL corresponding to the nodes to be grouped.
        
        Returns:
            list: list of cluster element objects.
        """
        with Cluster(cluster_name):
            return list(map(self.cn.create_node, node_names))

    def _create_cluster_params(self, cluster_name, node_names):
        """Creates a cluster for hyperparameter diagram with the given name and groups the given nodes to it.

        Args:
            cluster_name (str): Name of the cluster to be formed.
            node_names (list): two dimensional list containing the elements like Node Name, URL corresponding to the nodes to be grouped.
        
        """
        with self.g.subgraph(name='cluster_' + cluster_name) as c:
            inlabel = 'streamin_' + cluster_name
            outlabel = 'streamout_' + cluster_name
            c.attr(label=cluster_name, style='filled', color='cadetblue2', shape='record', nodesep="0.03")
            c.node(outlabel, label='Stream', shape='box', color='white')
            if cluster_name != 'Inputs':
                c.node(inlabel, label='Stream', shape='box', color='white')
                self.g.edge(self.input, inlabel)
                c.node(outlabel, label='Union', shape='box', color='white')
            for i in range(len(node_names)):
                c.node(cluster_name + str(i), label=node_names[i][0], URL=node_names[i][1])
                if cluster_name != 'Inputs':
                    c.edge(inlabel, str(cluster_name + str(i)))
                c.edge(cluster_name + str(i), outlabel)
            c.node_attr.update(style='filled', color='white', shape='record', nodesep='0.03')
            self.input = outlabel
