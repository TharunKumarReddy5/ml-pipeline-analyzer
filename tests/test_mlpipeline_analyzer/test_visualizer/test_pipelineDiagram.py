import diagrams
import graphviz
import joblib
from mlpipeline_analyzer.visualizer.PipelineDiagram import PipelineDiagram


class TestPipelineDiagram:
         
    def test_show_sklearn():
        sklearn_pipe = joblib.load('examples/sample_models/ml_pipeline.pkl')
        diagram = PipelineDiagram(sklearn_pipe)
        with diagrams.Diagram('', show=False, filename=''):
            assert diagrams.Diagram in diagram.show(title='Sklearn ML Pipeline Diagram').__class__.mro()
            
    def test_show_params_sklearn():
        sklearn_pipe = joblib.load('examples/sample_models/ml_pipeline.pkl')
        diagram = PipelineDiagram(sklearn_pipe)
        with diagrams.Diagram('', show=False, filename=''):
            assert graphviz.dot.Digraph in diagram.show_params(title='Sklearn Machine Learning Parameters Pipeline').__class__.mro()
    
    def test_show_evalml():
        evalml_pipe = joblib.load('examples/sample_models/automl_pipeline.pkl')
        diagram = PipelineDiagram(evalml_pipe)
        with diagrams.Diagram('', show=False, filename=''):
            assert diagrams.Diagram in diagram.show(title='Sklearn ML Pipeline Diagram').__class__.mro()
            
    def test_show_params_evalml():
        evalml_pipe = joblib.load('examples/sample_models/automl_pipeline.pkl')
        diagram = PipelineDiagram(evalml_pipe)
        with diagrams.Diagram('', show=False, filename=''):
            assert graphviz.dot.Digraph in diagram.show_params(title='Sklearn Machine Learning Parameters Pipeline').__class__.mro()

