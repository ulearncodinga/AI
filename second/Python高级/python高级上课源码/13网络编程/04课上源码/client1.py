import socket
import time
import threading


def func(id):
    # 创建一个 IPv4 的 TCP 套接字
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 定义服务器的地址和端口
    server_address = ('127.0.0.1', 43999)

    # 客户端连接到服务器的地址和端口
    client_socket.connect(server_address)

    message = str(client_socket.getpeername()) + str(id)
    while True:
        client_socket.send(message.encode())
        # 客户端接收来自服务器的最多 1024 字节的数据
        data = client_socket.recv(1024)
        # 打印接收到的来自服务器的响应消息
        print(f'来自服务器的响应为：{data.decode()}')
        # time.sleep(1)


thread_list = []
while True:
    for i in range(200):
        t = threading.Thread(target=func, args=(i, ))
        t.start()
        thread_list.append(t)
    for th in thread_list:
        th.join()