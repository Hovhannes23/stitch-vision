import io

import PIL
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from pillow_heif import register_heif_opener
from minio.error import InvalidResponseError
# import sys
# sys.path.append('/pythonChartService')
import pythonProject3.utils.engine as engine
import os
from minio import Minio

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heif'}
# для работы с heif форматом
register_heif_opener()


@app.route('/recognition/border', methods=['GET'])
# @swag_from("swagger/image_controller_api_doc.yml")
def get_stitch_border():

    object_name = request.json['imageId']
    bucket_name = 'task-images'
    minioClient = Minio(endpoint="192.168.1.181:9000", access_key= 'stitch', secret_key='stitch2023', secure=False)

    try:
        response = minioClient.get_object(bucket_name, object_name)
        image = Image.open(BytesIO(response.data))
    except InvalidResponseError as err:
        print("error", err)
    else:
        response.close()
        response.release_conn()

    if image.format.lower() not in ALLOWED_EXTENSIONS:
        resp = jsonify({'message': 'File type is not allowed'})
        resp.status_code = 400

    image = np.asarray(image)

    # определяем границы листа А4
    corner_pts_A4 = engine.detect_corner_points(image)
    # избавляемся от перспективного искажения на А4
    A4_no_distortion = engine.remove_perspective_distortion(image, corner_pts_A4, 0, 0)
    # сохраняем А4 в Minio
    A4_no_distortion = Image.fromarray(A4_no_distortion)
    bucket = 'recognized-corner'
    bucket_exists = minioClient.bucket_exists(bucket)
    if not bucket_exists:
        minioClient.make_bucket(bucket)

    b = BytesIO()
    A4_no_distortion.save(b, 'png')
    minioClient.put_object(bucket, object_name, io.BytesIO(b.getvalue()), b.getbuffer().nbytes, content_type='image/png')



    corner_pts = ''
    return corner_pts

@app.route('/support/image/clusterize', methods=['POST'])
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

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')