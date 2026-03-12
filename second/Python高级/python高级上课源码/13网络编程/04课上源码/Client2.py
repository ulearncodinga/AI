import socket
import time

# 创建一个 IPv4 的 TCP 套接字
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 定义服务器的地址和端口
server_address = ('127.0.0.1', 43999)

# 客户端连接到服务器的地址和端口
client_socket.connect(server_address)

print('已成功连接服务器...')
i = 0

while True:
    if i == 3:
        # 客户端向服务器发送经过编码的消息“我是客户端”
        client_socket.send('the last one'.encode())
    else:
        client_socket.send(str(i).encode())
    # 客户端接收来自服务器的最多 1024 字节的数据
    data = client_socket.recv(1024)
    # 打印接收到的来自服务器的响应消息
    print(f'来自服务器的响应为：{data.decode()}')
    if data.decode() == 'ok':
        break
    i += 1
    time.sleep(1)

# 关闭客户端套接字
client_socket.close()