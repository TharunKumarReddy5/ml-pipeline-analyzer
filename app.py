from flask import Flask, render_template, request, url_for, redirect
import os
import time
from mlpipeline_analyzer.visualizer import PipelineDiagram
import joblib

app = Flask(__name__)


@app.route('/')
def get_app():
    return render_template('index.html', view_id='')


@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '' and uploaded_file.filename.endswith('pkl'):
        fp = os.path.join('./uploads', 'model' + str(time.time()) + '.pkl')
        uploaded_file.save(fp)
        PipelineDiagram(joblib.load(fp))
    return render_template('index.html', view_id='tryit')


if __name__ == '__main__':
    app.run()
