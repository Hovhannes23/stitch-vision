import io
import json
import PIL
from PIL import Image
from flask import Flask, request, jsonify
from minio import Minio
from pillow_heif import register_heif_opener

import pythonProject3.service.support_service as support_service
import pythonProject3.utils.engine as engine

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heif'}
# для работы с heif форматом
register_heif_opener()


@app.route('/recognition/border', methods=['GET'])
# @swag_from("swagger/image_controller_api_doc.yml")
def get_stitch_border():

    object_name = request.json['imageId']
    bucket_to_get = 'task-images'
    minioClient = Minio(endpoint="192.168.1.181:9000", access_key= 'stitch', secret_key='stitch2023', secure=False)

    # получаем изображение из Minio
    image = support_service.get_object_from_minio(object_name, bucket_to_get, minioClient)
    # определяем границы листа А4
    corner_pts_A4 = engine.detect_corner_points(image)
    # избавляемся от перспективного искажения на А4
    A4_no_distortion = engine.remove_perspective_distortion(image, corner_pts_A4, 0, 0)
    # сохраняем изображение в Minio
    bucket_to_put = 'recognized-corner'
    A4_no_distortion = Image.fromarray(A4_no_distortion)
    object_name = support_service.put_object_to_minio(A4_no_distortion, object_name, bucket_to_put, minioClient, 'image/png')

    corner_pts_A4 = support_service.order_points(corner_pts_A4)
    response = {
        "id": object_name,
        "corners": {
            "leftTopCorner": corner_pts_A4[0].tolist(),
            "rightTopCorner": corner_pts_A4[1].tolist(),
            "rightDownCorner": corner_pts_A4[2].tolist(),
            "leftDownCorner": corner_pts_A4[3].tolist()
        }
    }

    resp = app.response_class(
        response=json.dumps(response),
        status=200,
        mimetype="application/json"
    )

    return resp

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