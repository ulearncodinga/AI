import json
# 定义一个字典
data_dict = {
    "username":"张三",
    "age":18,
    "gender":"男"
}
# 将字典转换成一个JSON文件 010001010
with open('data', 'w', encoding='utf-8') as f:
    json.dump(data_dict,f,ensure_ascii=False)