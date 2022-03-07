from sklearn.svm import SVC
from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.discriminant_analysis import *
from sklearn.impute import *
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from mlpipeline_analyzer import PipelineDiagram
import joblib


def custom_function(set=1):
    s = 'Hello' * set
    return


model = SVC(C=1.0, kernel='poly', degree=5, gamma='scale')
sklearn_pipeline = Pipeline(
    [('custom', custom_function(set=10)), ('labelencoder', LabelEncoder()),  # -- Pipe Transformer 1
     ("imputer", SimpleImputer(missing_values=np.nan, strategy='mean')),  # -- Pipe Transformer 2
     ('scale', FeatureUnion([
         ('minmax', MinMaxScaler()),  # -- Parallel Transformer 3
         ('standardscaler', StandardScaler()),  # -- Parallel Transformer 4
         ('normalize', Normalizer())])),  # -- Parallel Transformer 5
     ('feature_select', RFE(estimator=model, n_features_to_select=1)),  # -- Pipe Transformer 6
     ('PCA', PCA(n_components=1)),  # -- Pipe Transformer 7
     ("LDA", LinearDiscriminantAnalysis()),  # -- Pipe Transformer 8
     # ('classifier', model), 	      #-- Pipe Classifier/Predictor 9
     ('voting', RandomForestClassifier(n_estimators=10))])  # -- Pipe Classifier/Predictor 10

joblib.dump(sklearn_pipeline, 'sample_models/ml_pipeline.pkl')

sklearn_pipeline = joblib.load('sample_models/ml_pipeline.pkl')
a = PipelineDiagram(sklearn_pipeline)
a.show(title='Sklearn ML Pipeline Diagram')
a.show_params(title='Sklearn Machine Learning Parameters Pipeline')

evalml_pipeline = joblib.load('sample_models/automl_pipeline.pkl')
b = PipelineDiagram(evalml_pipeline)
b.show(title='Evalml ML Pipeline Diagram')
b.show_params(title='Evalml Machine Learning Parameters Pipeline')
