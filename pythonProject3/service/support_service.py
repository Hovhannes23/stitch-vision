from io import BytesIO
import numpy as np
from PIL import Image
from flask import jsonify
from minio.error import InvalidResponseError
from pillow_heif import register_heif_opener
import pythonProject3.utils.engine as engine

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heif'}
# для работы с heif форматом
register_heif_opener()

def get_object_from_minio(object_name, bucket_name, minioClient):

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
    return image

def put_object_to_minio(object, object_name, bucket_name, minioClient, content_type):

    bucket_exists = minioClient.bucket_exists(bucket_name)
    if not bucket_exists:
        minioClient.make_bucket(bucket_name)

    # переводим объект в поток bytes, чтобы сохранить в Minio
    bytes = BytesIO()
    object.save(bytes, 'png')
    minioClient.put_object(bucket_name, object_name, BytesIO(bytes.getvalue()), bytes.getbuffer().nbytes, content_type)

    return object_name

def order_points(pts):
    pts = engine.order_points(pts)
    return pts