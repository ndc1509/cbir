import json
import redis


# Tạo kết nối đến DB
def connectDB():
    r = redis.Redis(
        host='localhost',
        port=6379,
        password='',
        decode_responses=True
    )
    r.ping()
    print('DB connected')
    return r


# Lưu vector dưới dạng hash như sau
# Key: "images:idx"
# Value:
# {
#     "filename": "Tên file ảnh",
#     "vector": "Vector đặc trưng được lưu dưới dạng JSON string"
# }

def save_vectors_JSON(vector_dict):
    r = connectDB()
    p = r.pipeline(transaction=False)

    for idx, image_name in enumerate(vector_dict):
        value = {}
        key = "images:" + str(idx)
        value['filename'] = image_name
        value['vector'] = json.dumps(vector_dict[image_name].tolist())
        p.hset(key, mapping=value)
    p.execute()
    print("Vectors saved")


# lấy vector từ csdl
def get_vectors():
    r = connectDB()
    keys = r.keys("images:*")
    vectors = []
    for key in keys:
        data = r.hgetall(key)
        vectors.append(data)
    return vectors
