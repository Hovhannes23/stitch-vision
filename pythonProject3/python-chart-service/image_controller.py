# -*- coding: utf-8 -*-
# from flasgger import Swagger, swag_from
import json
import PIL
import numpy
from flask import Flask, request, jsonify
from PIL import Image
import io
import engine
import utils
from pillow_heif import register_heif_opener

app = Flask(__name__)
# swagger = Swagger(app)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heif'}
# для работы с heif форматом
register_heif_opener()


def allowed_file(filename):
   return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/image/upload', methods=['POST'])
# @swag_from("swagger/image_controller_api_doc.yml")
def upload_image():
    image_bytes = request.get_data()
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # удалить print после тестирования
        print("_________________________________")
        print("height:" + str(image.height))
        print("width:" + str(image.width))
        print("_________________________________")
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

        image = numpy.array(image)
        utils.showImage(image)
        labels, label_image_map = engine.get_cells_from_image(image, clusters_num, rows_num, columns_num)
        response = {}
        labels = labels.reshape(rows_num, columns_num)
        # labels_base64 = engine.encode_base64(labels)
        response["matrix"] = labels.tolist()
        response['symbolsMap'] = label_image_map

        response = engine.response_adapter(response, rows_num, columns_num)
        resp = app.response_class(
           response=json.dumps(response),
           status=202,
           mimetype='application/json'
        )
        return resp

# @app.route('/fonts', methods=['GET'])
# def get_fonts_from_file():
#     font_service.get_fonts_from_file("python-chart-service/fonts/CSTITCTG.ttf")


if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')
