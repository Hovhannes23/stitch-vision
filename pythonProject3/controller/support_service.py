import shutil
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from flask import jsonify
from kafka import KafkaConsumer
from minio import Minio
from minio.error import InvalidResponseError
from pillow_heif import register_heif_opener
from time import sleep
from dotenv import load_dotenv
from kafka import KafkaProducer
import json
import engine
import util
import os

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heif'}
# для работы с heif форматом
register_heif_opener()
load_dotenv()

def get_stitch_corner_pts(object_name, bucket_to_get, minio_client):

    # получаем изображение из Minio
    image =  get_object_from_minio(object_name, bucket_to_get, minio_client)
    # определяем границы листа А4
    corner_pts_A4 = engine.detect_corner_points(image)
    # избавляемся от перспективного искажения на А4
    A4_no_distortion = engine.remove_perspective_distortion(image, corner_pts_A4, 0, 0)
    # сохраняем изображение в Minio
    bucket_to_put = "recognized-corner"
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

def split_cells_and_archieve():
    global image_id
    value_serializer = lambda m: json.dumps(m).encode("utf-8")
    bootstrap_servers = ["0.0.0.0:9092"]
    producer = KafkaProducer(
        value_serializer=value_serializer,
        bootstrap_servers=bootstrap_servers
    )

    producer.send('topic1', value={
	"id": "12345",
	"sizeWidth": 52,
	"sizeHeight": 58,
	"symbols": 6,
	"backStitch": True,
	"frenchKnot": True,
	"image": {
		"imageId": "123456789",
		"leftTopCorner": [149, 131],
		"rightTopCorner": [1193, 129],
		"rightDownCorner": [1189, 1280],
		"leftDownCorner": [148, 1282]
	}
     })


    consumer = KafkaConsumer(
        'topic1',
        bootstrap_servers=["0.0.0.0:9092"],
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="consumer_group_id",
    )
    for message in consumer:
        message = json.loads(message.value.decode('utf8'))
        image_data = message["image"]
        image_id = image_data["imageId"]
        leftTopCorner = image_data["leftTopCorner"]
        rightTopCorner = image_data["rightTopCorner"]
        rightDownCorner = image_data["rightDownCorner"]
        leftDownCorner = image_data["leftDownCorner"]
        rows = message["sizeHeight"]
        columns = message["sizeWidth"]
        symbols = message['symbols']

        minioClient = Minio(endpoint=os.getenv('MINIO_ENDPOINT'), access_key=os.getenv('MINIO_ACCESS_KEY'),
                            secret_key=os.getenv('MINIO_SECRET_KEY'), secure=False)

        image = get_object_from_minio(image_id, "recognized-corner", minioClient)
        corner_pts = np.array([[[leftTopCorner[0], leftTopCorner[1]]],
                               [[rightTopCorner[0], rightTopCorner[1]]],
                               [[rightDownCorner[0], rightDownCorner[1]]],
                               [[leftDownCorner[0], leftDownCorner[1]]]
                               ])
        image = engine.remove_perspective_distortion(image,corner_pts,rows,columns)
        image = 255 - image
        cells = util.split_into_cells(image,rows,columns)
        labels, symbol_list = engine.cluster_cells(cells, symbols)
        root_path = util.get_project_root()
        image_directory = str(Path(root_path, "resources", "cell-images", image_id))
        for idx, cell in  enumerate(cells):
            label = labels[idx]
            cell_directory = image_directory + "/" + str(label)
            if not os.path.exists(cell_directory):
                os.makedirs(cell_directory)
            cell_coordinates = util.get_cell_coordinates(idx, columns)
            filename = str(cell_coordinates[0]) + '_' + str(cell_coordinates[1])
            engine.save_file(cell, cell_directory, filename)

        # создать архив и вернуть его, чтобы сохранить в minio
        shutil.make_archive(image_directory, 'zip', str(Path(root_path, "resources", "cell-images")), image_id)
        # переделать метод так, чтобы он не только png умел сохранять
        put_object_to_minio()

        # после сохранения всех cells архивировать папку resources/cells/image_id и сохранить в Minio в бакете clusterised-images


if __name__ == '__main__':
   split_cells_and_archieve()
