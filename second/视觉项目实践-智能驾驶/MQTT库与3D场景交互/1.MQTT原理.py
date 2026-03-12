'''QoS 0：最多一次（At most once）。消息不进行确认，可能会丢失。
QoS 1：至少一次（At least once）。消息至少传递一次，但可能会重复。
QoS 2：只有一次（Exactly once）。消息保证只传递一次，确保不重复。

MQTT（Message Queuing Telemetry Transport）是一种轻量级的、基于发布/订
阅模式的消息传输协议，专为低带宽、不可靠网络环境下的设备通信设计，尤其适用
于物联网（IoT）设备之间的通信，基于TCP/IP的。'''