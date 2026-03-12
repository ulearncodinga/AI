import socket
import selectors

sel = selectors.DefaultSelector()

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
address = ('127.0.0.1', 43999)
server_socket.bind(address)
server_socket.listen(5)

print('服务器已启动...')

sel.register(server_socket, selectors.EVENT_READ, data=None)

while True:
    events = sel.select()
    for key, mask in events:
        if key.fileobj is server_socket:
            connection, client_address = server_socket.accept()
            print(f'接受来自 {client_address} 的连接')
            connection.setblocking(False)
            sel.register(connection, selectors.EVENT_READ, data=client_address)
        else:
            try:
                sock = key.fileobj
                data = sock.recv(1024)
                if not data:
                    print(f'{sock.getpeername()}已断开连接')
                    sel.unregister(sock)
                    sock.close()
                else:
                    print(f'接收来自{sock.getpeername()}的数据：{data.decode()}')
                    response = '你发过来的数据是' + data.decode()
                    sock.send(response.encode())
            except Exception as e:
                print(e)
                sel.unregister(sock)
                sock.close()