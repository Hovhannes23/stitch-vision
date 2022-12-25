import json
import PIL
import numpy
from flask import Flask, request, jsonify
from PIL import Image
import io
from pillow_heif import register_heif_opener
# import sys
# sys.path.append('/pythonChartService')
import pythonProject3.pythonChartService.engine as engine
import pythonProject3.pythonChartService.utils as utils

app = Flask(__name__)

@app.route('/image/upload', methods=['POST'])
# @swag_from("swagger/image_controller_api_doc.yml")
def upload_image():
    return ""