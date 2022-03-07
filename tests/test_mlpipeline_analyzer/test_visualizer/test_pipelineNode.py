import diagrams
from mlpipeline_analyzer.visualizer.PipelineNode import PipelineNode


class TestPipelineNode:
    def test_create_node_data(self):
        node = PipelineNode()
        with diagrams.Diagram('', show=False, filename=''):
            assert diagrams.Node in node.create_node(['data', 'data']).__class__.mro()

    def test_create_node_custom(self):
        node = PipelineNode()
        with diagrams.Diagram('', show=False, filename=''):
            assert diagrams.Node in node.create_node(['Custom Function', 'Custom Function']).__class__.mro()

    def test_create_node_stream(self):
        node = PipelineNode()
        with diagrams.Diagram('', show=False, filename=''):
            assert diagrams.Node in node.create_node(['Data Stream', 'Data Stream']).__class__.mro()

    def test_create_node_preprocessing(self):
        node = PipelineNode()
        with diagrams.Diagram('', show=False, filename=''):
            assert diagrams.Node in node.create_node(['preprocessing', 'preprocessing']).__class__.mro()

    def test_create_node_feature_selection(self):
        node = PipelineNode()
        with diagrams.Diagram('', show=False, filename=''):
            assert diagrams.Node in node.create_node(['feature_selection', 'feature_selection']).__class__.mro()

    def test_create_node_cross_decomposition(self):
        node = PipelineNode()
        with diagrams.Diagram('', show=False, filename=''):
            assert diagrams.Node in node.create_node(['cross_decomposition', 'cross_decomposition']).__class__.mro()

    def test_create_node_cluster(self):
        node = PipelineNode()
        with diagrams.Diagram('', show=False, filename=''):
            assert diagrams.Node in node.create_node(['cluster', 'cluster']).__class__.mro()

    def test_create_node_ensemble(self):
        node = PipelineNode()
        with diagrams.Diagram('', show=False, filename=''):
            assert diagrams.Node in node.create_node(['ensemble', 'ensemble']).__class__.mro()

    def test_create_node_model_selection(self):
        node = PipelineNode()
        with diagrams.Diagram('', show=False, filename=''):
            assert diagrams.Node in node.create_node(['model_selection', 'model_selection']).__class__.mro()

    def test_create_node_multiclass(self):
        node = PipelineNode()
        with diagrams.Diagram('', show=False, filename=''):
            assert diagrams.Node in node.create_node(['multiclass', 'multiclass']).__class__.mro()

    def test_create_node_metrics(self):
        node = PipelineNode()
        with diagrams.Diagram('', show=False, filename=''):
            assert diagrams.Node in node.create_node(['metrics', 'metrics']).__class__.mro()

    def test_create_node_impute(self):
        node = PipelineNode()
        with diagrams.Diagram('', show=False, filename=''):
            assert diagrams.Node in node.create_node(['impute', 'impute']).__class__.mro()
