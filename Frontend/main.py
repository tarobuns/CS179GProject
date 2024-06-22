from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory,jsonify
import os
from readDB import fetch_data_from_mysql
from werkzeug.utils import secure_filename
from PIL import Image
from pyo_classification import Predict
from readDB import getNameAndHDFSurl
# Initialize the Flask application
app = Flask(__name__)


# Set the upload folder and allowed extensions
UPLOAD_FOLDER = '/home/cs179g/code/Demo/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey'

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    


@app.route('/')
def index():
    return render_template('main.html')

@app.route('/start',methods=['POST'])
def start():
    return render_template('main.html')




@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    filename = None
    if request.method == 'GET':
        try:
            # Extract value from the JSON request body
            value = request.args.get('value')
            # Convert the value to an integer
            index = int(value)
            path,name = getNameAndHDFSurl(index)
            return render_template('upload.html',path=path,name=name,index=index)
        except ValueError:
            return jsonify({'error': 'Invalid integer value'}), 400
        except (TypeError, KeyError):
            return jsonify({'error': 'Value parameter is missing'}), 400

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url,path=path,name=name)
        file = request.files['file']
        index = request.form['index']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url,path=path,name=name)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = Image.open(file)
            img = img.resize((224, 224))
            img.save(filepath)
            result=Predict(img,int(index))
            flash('File successfully uploaded')
            return render_template('upload.html', filename=filename,result=result)
    return render_template('upload.html', filename=filename)


@app.route('/button_click')
def button_click():
    # Simulate button click behavior (doesn't actually submit data)
    return redirect('/predict')  # Use code=302 for a temporary redirect



@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/models',methods=['GET'])
def print_models():
    return render_template('models.html')

@app.route('/api')
def get_data():
    # Fetch data from MySQL and return as JSON
    data = fetch_data_from_mysql()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True,port=5555)
