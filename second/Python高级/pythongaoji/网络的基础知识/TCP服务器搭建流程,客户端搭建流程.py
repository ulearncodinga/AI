
"""

服务器搭建
"""

import socket
#1.创建套接字
server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#2.bind绑定,用来将网络服务绑定到指定的IP地址和端口号
server_address = ('127.0.0.1',14165)
server_socket.bind(server_address)
#3.listen(backlog)监听,必须在绑定之后去进行,用来监听客户端的连接请求
server_socket.listen(5)
#4.accpet(),会返回两个参数:第一个是连接上来的客户端的套接字,第二个是连接上来的客户端的IP地址
client_socket,client_address = server_socket.accept()
#接收客户端发过来的消息
data = client_socket.recv(1024)
#收到客户端发来的消息后,对消息做一个处理
data_decode = data.decode()

message = '我收到你的消息是: '+ data_decode
#对消息处理完之后,对客户端进行一个请求的反馈
client_socket.send(message.encode())

#在没有任何消息传来的时候,或者不需要服务器的时候可以使用close去关闭服务器
server_socket.close()

# C/S架构
# client:客户端
# Server:服务器




import socket

#创建套接字
client_socket = socket.socket(socket.AF.INET,socket.SOCK_STREAM)

#2.使用connect去连接到服务器上
server_address= ('127.0.0.1',14165)
client_address.connect(server_address)

#3作为客户端,通常是发起请求的一方,也就是说客户端通常需要先去发送请求
client_socket.send('我是客户端'.encode())

#4在向服务器发送请求后,需要监听发送的反馈信息
data=client_socket.recv(1024)
data_decode = data.decode()

print('服务器返回来的内容',data_decode)

#5关闭客户端
client_socket.close()



























