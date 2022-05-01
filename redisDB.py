import json

import numpy
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.query import Query


# Tạo kết nối đến DB
def connectDB():
    r = redis.Redis(
        host='localhost',
        port=6379,
        password=''
    )
    r.ping()
    print('DB connected')
    return r


# Cách 1: Sử dụng chức năng tìm kiếm vector tương đồng RediSearch
# Lưu dữ liệu dưới dạng hash như sau:
# Key: int
# Value:
# {
#     "filename": "Tên file ảnh",
#     "vector": "Chuỗi byte biểu diễn đặc trưng ảnh"
# }
def save_vectors(vector_dict):
    r = connectDB()
    # Kỹ thuật pipeline giúp tăng hiệu năng bằng cách
    # thực thi nhiều lệnh một lúc mà không cần chờ phản hồi
    # https://redis.io/docs/manual/pipelining/
    p = r.pipeline(transaction=False)

    # Duyệt tuần tự từ điển vector_dict
    for idx, image_name in enumerate(vector_dict):
        value = {}
        key = idx
        value['filename'] = image_name
        value['vector'] = vector_dict[image_name].astype(numpy.float32).tobytes()
        # print(len(value['vector'])) # Độ dài chuỗi byte 36496
        # Tại sao là 36492 ?
        # Vì 1 chiều vector biểu diễn dưới dạng float32 => 4 bytes
        # Vector có 9124 chiều => 32 x 4 = 36496 bytes
        p.hset(key, mapping=value)
    p.execute()
    print("Vectors saved")


# Tạo index FLAT https://milvus.io/docs/v1.0.0/index.md#FLAT
def create_flat_index():
    connectDB().ft().create_index([
        TextField("filename"),
        VectorField("vector", "FLAT", {
            "TYPE": "FLOAT32",
            "DIM": 9124,  # Số chiều của vector
            "DISTANCE_METRIC": 'L2',  # Sử dụng khoảng cách Euclid để so sánh sự tương đồng
            "INITIAL_CAP": 100,  # Option số lượng vector cần lưu trữ
            "BLOCK_SIZE": 100
        }),
    ])
    print("Index created")


# Tạo câu truy vấn
# input vector: chuỗi byte biểu diễn vector đặc trưng ảnh đầu vào
# return 10 ảnh (Hiện tại là 3)
def query(vector):
    q = Query(f'*=>[KNN 3 @vector $vec_param AS vector_score]').sort_by('vector_score').return_fields(
        "vector_score", "filename").dialect(2)
    params_dict = {"vec_param": vector}
    results = connectDB().ft().search(q, query_params=params_dict)
    return results


# Xóa index
def drop_index():
    connectDB().ft().dropindex(True)
    print("Index dropped")


# Cách 2 Tính toán sự tương đồng bằng thư viện scipy
# Lưu vector dưới dạng hash như sau
# Key: int
# Value:
# {
#     "filename": "Tên file ảnh",
#     "vector": "Vector đặc trưng được lưu dưới dạng JSON"
# }

def save_vectors_JSON(vector_dict):
    r = connectDB()
    p = r.pipeline(transaction=False)

    for idx, image_name in enumerate(vector_dict):
        value = {}
        key = "image:" + str(idx)
        value['filename'] = image_name
        value['vector'] = json.dumps(vector_dict[image_name].tolist())
        p.hset(key, mapping=value)
    p.execute()
    print("Vectors saved")


def get_vectors_JSON():
    r = connectDB()
    vector_dict = r.hgetall("image:*")
    print(vector_dict)


# get_vectors_JSON()