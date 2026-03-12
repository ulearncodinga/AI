''''''
'''
MQTT（Message Queuing Telemetry Transport）是一种轻量级的、基于发布/订
阅模式的消息传输协议，专为低带宽、不可靠网络环境下的设备通信设计，尤其适用于物联网
（IoT）设备之间的通信，基于TCP/IP的。

MQTT架构:
客户端(Client)
代理(Broker)
主题(Topic)

QOS代表发送信息的质量
又 0 1 2三种
QoS 0：最多一次（At most once）。消息不进行确认，可能会丢失。
QoS 1：至少一次（At least once）。消息至少传递一次，但可能会重复。
QoS 2：只有一次（Exactly once）。消息保证只传递一次，确保不重复。

发布/订阅模式:

发布者（Publisher）：发布者向代理发布消息，并指定消息的主题。
订阅者（Subscriber）：订阅者向代理订阅一个或多个主题，以接收相应的消
息。
代理（Broker）：代理接收发布者的消息，并将其分发给订阅了相应主题的订阅
者。
'''