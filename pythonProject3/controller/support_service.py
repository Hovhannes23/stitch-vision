import re
import shutil
import zipfile
from io import BytesIO
from pathlib import Path

import cv2
import kafka
import numpy as np
from PIL import Image
from kafka import KafkaConsumer
from kafka.admin import NewTopic
from minio import Minio
from minio.error import InvalidResponseError
from pillow_heif import register_heif_opener
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
minio_client = Minio(endpoint=os.getenv('MINIO_ENDPOINT'), access_key=os.getenv('MINIO_ACCESS_KEY'),
                                secret_key=os.getenv('MINIO_SECRET_KEY'), secure=False)
root_path = util.get_project_root()

def get_stitch_corner_pts(object_name, bucket_to_get, minio_client):

    # получаем изображение из Minio
    image =  get_object_from_minio(object_name, bucket_to_get, minio_client)
    # определяем границы листа А4
    corner_pts_A4 = engine.detect_corner_points(image)
    # избавляемся от перспективного искажения на А4
    A4_no_distortion_array = engine.remove_perspective_distortion(image, corner_pts_A4, 0, 0)
    # сохраняем изображение в Minio
    bucket_to_put = "recognized-corner"
    A4_no_distortion = Image.fromarray(A4_no_distortion_array)
    bytes = BytesIO()
    A4_no_distortion.save(bytes, 'png')
    object_name = put_object_to_minio(bytes, bucket_to_put, object_name, minio_client, 'image/png')

    # находим границы эскиза
    A4_no_distortion_array = 255 - A4_no_distortion_array
    corner_pts_stitch = engine.detect_corner_points(A4_no_distortion_array)
    corner_pts_stitch = corner_pts_stitch.squeeze()
    corner_pts = {
        "id": object_name,
        "corners": {
            "leftTopCorner": corner_pts_stitch[0].tolist(),
            "rightTopCorner": corner_pts_stitch[1].tolist(),
            "rightDownCorner": corner_pts_stitch[2].tolist(),
            "leftDownCorner": corner_pts_stitch[3].tolist()
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
        raise ValueError("FIle type %s is not allowed" % image.format.lower())

    image = np.asarray(image)
    return image

def put_object_to_minio(object_bytes, bucket_name, object_name,  minio_client, content_type):

    bucket_exists = minio_client.bucket_exists(bucket_name)
    if not bucket_exists:
        minio_client.make_bucket(bucket_name)

    minio_client.put_object(bucket_name, object_name, BytesIO(object_bytes.getvalue()), object_bytes.getbuffer().nbytes, content_type)

    return object_name

def fput_object_to_minio(bucket_name, object_name, object_path, minio_client, content_type):
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
    minio_client.fput_object( bucket_name, object_name, object_path, content_type)

def order_points(pts):
    pts = engine.order_points(pts)
    return pts

def split_cells_and_archive():
    # logging.info("split_cells_and_archive started")
    global image_id
    value_serializer = lambda m: json.dumps(m).encode("utf-8")
    bootstrap_servers = [os.getenv('KAFKA_ENDPOINT')]
    # bootstrap_servers = ["localhost:9092"]
    root_path = util.get_project_root()
    ssl_cafile = str(Path(root_path, "controller", "CARoot.pem"))
    ssl_certfile = str(Path(root_path, "controller", "certificate.pem"))
    ssl_keyfile = str(Path(root_path, "controller", "key.pem"))

    producer = KafkaProducer(
        value_serializer=value_serializer,
        bootstrap_servers=bootstrap_servers,
        api_version_auto_timeout_ms= 20000,
        security_protocol="SSL",
        ssl_check_hostname = True,
        ssl_cafile = ssl_cafile,
        ssl_certfile = ssl_certfile,
        ssl_keyfile= ssl_keyfile,
        ssl_password="changeit"
    )

    admin_kafka_client = kafka.KafkaAdminClient(
        bootstrap_servers=bootstrap_servers,
        client_id='stitch-vision',
        security_protocol="SSL",
        ssl_check_hostname = True,
        ssl_cafile = ssl_cafile,
        ssl_certfile = ssl_certfile,
        ssl_keyfile= ssl_keyfile,
        ssl_password="changeit"
    )

    producer.send('recognition', value={
	"id": "123456789",
	"sizeWidth": 52,
	"sizeHeight": 58,
	"symbols": 6,
	"backStitch": True,
	"frenchKnot": True,
	"image": {
		"id": "123456789",
        "corners": {
            "leftTopCorner": [149, 131],
            "rightTopCorner": [1193, 129],
            "rightDownCorner": [1189, 1280],
            "leftDownCorner": [148, 1282]
        }
	}
     })

    producer.flush()

    consumer = KafkaConsumer(
        'recognition',
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id="stitch_vision_group_id",
        security_protocol="SSL",
        ssl_check_hostname = True,
        ssl_cafile = ssl_cafile,
        ssl_certfile = ssl_certfile,
        ssl_keyfile= ssl_keyfile,
        ssl_password="changeit"
    )

    topic_to_send = "stitch.vision.archive.before.check"
    if topic_to_send not in admin_kafka_client.list_topics():
        admin_kafka_client.create_topics([NewTopic(name=topic_to_send, num_partitions=1, replication_factor=1)])
    for message in consumer:
        try:
            # парсим message
            message = json.loads(message.value.decode('utf8'))
            # logging.info("start to consume message with task_id: " + message["id"])
            task_id = message["id"]
            image_data = message["image"]
            image_id = image_data["id"]
            corners = image_data["corners"]
            leftTopCorner = corners["leftTopCorner"]
            rightTopCorner = corners["rightTopCorner"]
            rightDownCorner = corners["rightDownCorner"]
            leftDownCorner = corners["leftDownCorner"]
            rows = message["sizeHeight"]
            columns = message["sizeWidth"]
            symbols = message['symbols']

            # достаем изображение из Minio
            # logging.info("start to get image from Minio. Image_id: " + message)
            image = get_object_from_minio(image_id, "recognized-corner", minio_client)

            # избавляемся от перспективного искажения
            corner_pts = np.array([[[leftTopCorner[0], leftTopCorner[1]]],
                                   [[rightTopCorner[0], rightTopCorner[1]]],
                                   [[rightDownCorner[0], rightDownCorner[1]]],
                                   [[leftDownCorner[0], leftDownCorner[1]]]
                                   ])
            # image = 255 - image
            image = engine.remove_perspective_distortion(image, corner_pts, rows, columns)

            # вырезаем ячейки
            # image = 255 - image
            cells = util.split_into_cells(image,rows,columns)

            # кластеризируем
            labels, symbol_list = engine.cluster_cells(cells, symbols)

            # сохраняем ячейки в папки
            # logging.info("saving")

            image_directory = str(Path(root_path, "resources", "cell-images", image_id))
            for idx, cell in  enumerate(cells):
                label = labels[idx]
                cell_directory = image_directory + "/" + str(label)
                if not os.path.exists(cell_directory):
                    os.makedirs(cell_directory)
                cell_coordinates = util.get_cell_coordinates(idx, columns)
                filename = str(cell_coordinates[0]) + '_' + str(cell_coordinates[1])
                engine.save_file(cell, cell_directory, filename)

            # создаем архив и сохраняем в Minio
            archive_path = shutil.make_archive(base_name=image_directory, format= 'zip',
                                                root_dir=str(Path(root_path, "resources", "cell-images")),
                                                base_dir= image_id)
            bucket_name = "archives"
            object_name = str(task_id) + ".zip"
            content_type =  "application/zip"
            fput_object_to_minio(bucket_name, object_name, archive_path, minio_client, content_type)

            # в Kafka отправляем сообщение о готовности архива
            message_to_kafka = {
                "taskId": task_id,
                "success": True
            }

            send_message_to_kafka(message_to_kafka, producer, topic_to_send)
            consumer.commit()

            # удаляем папку и архив с изображениями ячеек
            shutil.rmtree(image_directory)
            Path(archive_path).unlink()
        except Exception as e:
            print(e)
            continue


def send_message_to_kafka(message, producer, topic_name):
    producer.send(topic_name, message)
    producer.flush()

def archive_to_json_response(id, folder_unicode_map, rows, columns):
    # архив достаем из Minio, разархивируем и сохраняем
    response = minio_client.get_object("correct-archives", id + ".zip")
    archive = BytesIO(response.data)
    archive = zipfile.ZipFile(archive)
    path_to_unpack = Path(root_path, "resources", "correct-archive-unpacked")
    archive.extractall(path_to_unpack)

    # парсим json response
    response = {
        "rows": rows,
        "columns": columns
    }
    symbols = []

    # итерируемся по папкам архива и парсим данные для response
    for dir_name in os.listdir(Path(path_to_unpack, id)):
        # достаем первое изображение, чтобы  определить цвет фона символа
        image_name = os.listdir(Path(path_to_unpack, id, dir_name))[0]
        image = cv2.imread(str(Path(path_to_unpack, id, dir_name, image_name)))
        # image = Image.open(Path(path_to_unpack, id, dir_name, image_name))
        color = engine.detach_background(image)[1]
        symbol_data = {
            "index": int(dir_name),
            "symbol": folder_unicode_map[dir_name],
            "color": color
        }
        coordinates = []

        for image_name in os.listdir(Path(path_to_unpack, id, dir_name)):
            row_column = re.split(r'[_,.]', image_name)
            # row_column = image_name.split(("_", "."))
            coordinates.append({
                "row": int(row_column[0]) - 1,
                "column": int(row_column[1]) - 1
            })

        symbol_data["coordinates"] = coordinates
        symbols.append(symbol_data)

    response["symbols"] = symbols
    return response

if __name__ == '__main__':
   split_cells_and_archive()
