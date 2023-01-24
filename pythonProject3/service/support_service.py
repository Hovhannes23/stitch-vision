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

def get_stitch_corner_pts(object_name, bucket_to_get, minio_client):

    # получаем изображение из Minio
    image =  get_object_from_minio(object_name, bucket_to_get, minio_client)
    # определяем границы листа А4
    corner_pts_A4 = engine.detect_corner_points(image)
    # избавляемся от перспективного искажения на А4
    A4_no_distortion = engine.remove_perspective_distortion(image, corner_pts_A4, 0, 0)
    # сохраняем изображение в Minio
    bucket_to_put = 'recognized-corner'
    A4_no_distortion = Image.fromarray(A4_no_distortion)
    object_name = put_object_to_minio(A4_no_distortion, object_name, bucket_to_put, minio_client, 'image/png')

    corner_pts_A4 = order_points(corner_pts_A4)
    corner_pts = {
        "id": object_name,
        "corners": {
            "leftTopCorner": corner_pts_A4[0].tolist(),
            "rightTopCorner": corner_pts_A4[1].tolist(),
            "rightDownCorner": corner_pts_A4[2].tolist(),
            "leftDownCorner": corner_pts_A4[3].tolist()
        }
    }
    return corner_pts

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

def put_object_to_minio(object, object_name, bucket_name, minio_client, content_type):

    bucket_exists = minio_client.bucket_exists(bucket_name)
    if not bucket_exists:
        minio_client.make_bucket(bucket_name)

    # переводим объект в поток bytes, чтобы сохранить в Minio
    bytes = BytesIO()
    object.save(bytes, 'png')
    minio_client.put_object(bucket_name, object_name, BytesIO(bytes.getvalue()), bytes.getbuffer().nbytes, content_type)

    return object_name

def order_points(pts):
    pts = engine.order_points(pts)
    return pts