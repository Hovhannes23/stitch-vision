from dotenv import load_dotenv
import io
import json
import PIL
from PIL import Image
from flask import Flask, request, jsonify
import support_service
import engine

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heif'}
load_dotenv()

# @app.route('/recognition/border', methods=['GET'])
def get_stitch_border(object_name, bucket_to_get, minioClient):

    corner_pts = support_service.get_stitch_corner_pts(object_name, bucket_to_get, minioClient)

    response = app.response_class(
        response=json.dumps(corner_pts),
        status=200,
        mimetype="application/json"
    )

    return response

# @app.route('/support/image/clusterize', methods=['POST'])
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