extends Node2D

@onready var zmq_receiver = ZMQReceiver.new_from("tcp://localhost:5555", 
	ZMQ.SocketType.PUB, ZMQ.ConnectionMode.CONNECT, "")

func _ready():
	pass
	#add_child(zmq_receiver)

	#zmq_receiver.onMessageString(func(s: String):
		#print("[ZMQ Receiver] Received: ", s)
	#)
	#zmq_receiver.onMessageBytes(func(bytes: PackedByteArray):
		#print("[ZMQ Receiver] Received: ", bytes.size())
	#)

func _exit_tree():
	zmq_receiver.stop()
	#remove_child(zmq_receiver)
