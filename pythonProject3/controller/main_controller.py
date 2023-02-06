# -*- coding: utf-8 -*-
# from flasgger import Swagger, swag_from
from threading import Thread

from dotenv import load_dotenv
from flask import Flask
from pillow_heif import register_heif_opener
import os
from dotenv import load_dotenv
import io
import json
import PIL
from PIL import Image
from flask import Flask, request, jsonify
from minio import Minio
import image_controller
import support_controller
from kafka import KafkaProducer
import support_service

# import pythonProject3.controller.image_controller as image_controller
# import pythonProject3.controller.support_controller as support_controller

app = Flask(__name__)
# swagger = Swagger(app)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heif'}
# для работы с heif форматом
register_heif_opener()
load_dotenv()

def allowed_file(filename):
   return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/image/upload', methods=['POST'])
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
        return resp

    if success:
        clusters_num = request.args.get('clusters')
        rows_num = int(request.args.get('rows'))
        columns_num = int(request.args.get('columns'))
    return image_controller.upload_image(image, clusters_num, rows_num, columns_num)


@app.route('/recognition/border', methods=['POST'])
def get_stitch_border():
    object_name = request.json['imageId']
    bucket_to_get = 'task-images'
    minioClient = Minio(endpoint=os.getenv('MINIO_ENDPOINT'), access_key=os.getenv('MINIO_ACCESS_KEY'),
                        secret_key=os.getenv('MINIO_SECRET_KEY'), secure=False)
    return support_controller.get_stitch_border(object_name, bucket_to_get, minioClient)

@app.route('/recognition/startSplitAndArchive', methods=['GET'])
def start_split_and_archive():
    support_service.split_cells_and_archive()

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')


