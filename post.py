import requests

url = "http://192.168.3.156:5000/parse_overtime"
data = {
    "text": "明天下午 8 点12到晚上 10 点 在深圳加班写代码"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=data, headers=headers)

# 打印结果
print(response.status_code)
print(response.json())
