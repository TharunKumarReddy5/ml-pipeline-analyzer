from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)


@app.route('/')
def get_app():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run()
