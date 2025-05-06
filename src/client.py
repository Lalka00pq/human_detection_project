import requests
load_models_responce = requests.get("http://127.0.0.1:8000/load_model")
print(load_models_responce.json())
# health_responce = requests.get("http://127.0.0.1:8000/health")
# print(health_responce.json())

# check_classes_info = requests.get("http://127.0.0.1:8000/get_classes_info")
# print(check_classes_info.json())

# path_to_image = r"src\photo_2025-04-14_19-54-00.jpg"
# images = {"image": open(path_to_image, "rb")}
# response = requests.post("http://127.0.0.1:8000/image_inference", files=images)
# print(response.json())
