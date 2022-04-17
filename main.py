from distutils.log import debug
from flask import *
from design_captcha import *
from trademark_captcha import *
from ecourts import *
import base64   
import tensorflow as tf
from waitress import serve

app = Flask(__name__)

@app.route('/ecourts', methods=['POST'])
def ecourts():
	captcha = request.files['captcha']
	filename = "captcha.png"
	captcha.save(filename)
	return ecourts_captcha(filename)

@app.route('/design', methods=['POST'])
def design():
	new_quark = request.get_json()
	imgdata = base64.b64decode(new_quark["captcha"])
	filename = 'design.png'
	with open(filename, 'wb') as f:
		f.write(imgdata)
	return design_captcha(filename)

@app.route('/trademark', methods=['POST'])
def trademark():
	file = request.files['image']
	filename = "captcha.jpg"
	file.save(filename)
	return trademark_captcha(filename)
	
if __name__ == '__main__':
	serve(app,host="0.0.0.0", port=5000)
