from flask import Flask, render_template, request
from flask_executor import Executor
import os
import time
from mlpipeline_analyzer.visualizer import PipelineDiagram
import joblib

app = Flask(__name__)
executor = Executor(app)
PIPELINE_ERROR_MSG = 'Oops! Something went wrong. Make sure the file you uploaded is in the right format and ' \
                     'contains a scikit-learn pipeline object, else report the issue.'


@app.route('/')
def get_app():
    return render_template('index.html', view_id='', pipeline_img='/static/img.png')


def file_clean_up(files):
    time.sleep(5)
    for fp in files:
        os.remove(fp)


@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    try:
        fname = str(time.time()) + '.pkl'
        fp = os.path.join(os.getcwd(), 'uploads', 'model' + fname)
        uploaded_file.save(fp)
        PipelineDiagram(joblib.load(fp), file_name=fname[:-4]).create_diagram()
        pipeline_img_nm = fname[:-3] + 'png'
        pipeline_img_dest = os.path.join(os.getcwd(), 'app', 'static', pipeline_img_nm)
        os.rename(pipeline_img_nm, pipeline_img_dest)
        executor.submit(file_clean_up, [fp, pipeline_img_dest])
        return render_template('index.html', view_id='tryit', pipeline_img='/static/' + fname[:-3] + 'png')
    except Exception as e:
        print("Exception occurred with file: " + fp)
        print(e)
        return render_template('index.html', view_id='tryit', pipeline_img='', alt_text=PIPELINE_ERROR_MSG)


if __name__ == '__main__':
    app.run()
