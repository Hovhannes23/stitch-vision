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
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heif'}
# для работы с heif форматом
register_heif_opener()
@app.route('support/image/upload', methods=['POST'])
# @swag_from("swagger/image_controller_api_doc.yml")
def upload_image():
    image_bytes = request.get_data()
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except PIL.UnidentifiedImageError as e:
        resp = jsonify({'message': 'No image in request'})
        resp.status_code = 400
        return resp

    if image.format.lower() in ALLOWED_EXTENSIONS:
        success = True
    else:
        resp = jsonify({'message': 'File type is not allowed'})
        resp.status_code = 400
    corner_pts = engine.detect_corner_points(image)

    return corner_pts

@app.route('support/image/clusterize', methods=['POST'])
# @swag_from("swagger/image_controller_api_doc.yml")
def clusterize_cells():
    image_bytes = request.get_data()
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except PIL.UnidentifiedImageError as e:
        resp = jsonify({'message': 'No image in request'})
        resp.status_code = 400
        return resp

    if image.format.lower() in ALLOWED_EXTENSIONS:
        success = True
    else:
        resp = jsonify({'message': 'File type is not allowed'})
        resp.status_code = 400

    # будем получать от фронта
    correct_corner_pts = []
    clusters_num = int(request.args.get('clusters'))
    rows_num = int(request.args.get('rows'))
    columns_num = int(request.args.get('columns'))

    engine.detect_and_get_cells_for_sup(image, correct_corner_pts, rows_num, columns_num, clusters_num)