import socket
import asyncio


async def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('127.0.0.1', 43999)
    client_socket.connect(server_address)
    client_socket.setblocking(False)
    print('已成功连接服务器...')

    loop = asyncio.get_running_loop()
    while True:
        try:
            await loop.sock_sendall(client_socket, '我是客户端'.encode())
            data = await loop.sock_recv(client_socket, 1024)
            print('从服务器返回的数据是：', data.decode())
            await asyncio.sleep(1)
        except Exception as e:
            print('客户端异常，连接已退出...')
            client_socket.close()
            break

asyncio.run(main())