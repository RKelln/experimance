class_name ZMQ
extends Node

enum SocketType {
	PUB = 1,
	SUB = 2,
	REQ = 3,
	REP = 4,
	DEALER = 5,
	ROUTER = 6,
	PULL = 7,
	PUSH = 8,
	XPUB = 9,
	XSUB = 10,
	STREAM = 11
}

enum ConnectionMode {
	BIND = 1,
	CONNECT = 2
}
