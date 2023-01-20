import PIL
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from pillow_heif import register_heif_opener
from minio.error import InvalidResponseError
# import sys
# sys.path.append('/pythonChartService')
import pythonProject3.utils.engine as engine
from minio import Minio

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heif'}
# для работы с heif форматом
register_heif_opener()


@app.route('/recognition/border', methods=['GET'])
# @swag_from("swagger/image_controller_api_doc.yml")
def get_stitch_border():

    bucket_name = 'task-images'
    object_name = '0263ce2b-03fb-4847-a47c-6a50d670253'
    minioClient = Minio(endpoint="192.168.1.181:9000", access_key= 'stitch', secret_key='stitch2023', secure=False)

    try:
        response = minioClient.get_object(bucket_name, object_name)
        image = Image.open(BytesIO(bytes(response.data)))
    except InvalidResponseError as err:
        print("error", err)
    else:
        response.close()
        response.release_conn()

    if image.format.lower() in ALLOWED_EXTENSIONS:
        success = True
    else:
        resp = jsonify({'message': 'File type is not allowed'})
        resp.status_code = 400
    corner_pts = engine.detect_corner_points(image)

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