class_name ZeroMQReceiver
extends Node

enum Mode {
	PUSH_PULL = 1,
	PUB_SUB = 2,
	REQ_REP = 3
}

# @export var zmq_test_mode:TestMode = TestMode.REQ_REP
@export var zmq_mode:Mode = Mode.PUB_SUB
@export var image_format:StringName = "png"
@export var target_size:Vector2i = Vector2i(1080,1080)
@export var interp_mode:int = Image.INTERPOLATE_LANCZOS

@onready var zmq_receiver:ZMQReceiver = null

var pending_topic:String = ""

func create_zmq_receiver():
	var zmq_in_address = get_zmq_in_address()
	var zmq_in_socket_type = get_zmq_in_socket_type()
	var zmq_in_connection_mode = get_zmq_in_connection_mode()

	return ZMQReceiver.new_from(zmq_in_address, zmq_in_socket_type, zmq_in_connection_mode, "")

func get_zmq_in_address():
	return "tcp://localhost:5555"

func get_zmq_out_address():
	return "tcp://localhost:5555"

func get_zmq_in_socket_type():
	match zmq_mode:
		Mode.PUSH_PULL:
			return ZMQ.SocketType.PULL
		Mode.PUB_SUB:
			return ZMQ.SocketType.SUB
		Mode.REQ_REP:
			return ZMQ.SocketType.REP

func get_zmq_out_socket_type():
	match zmq_mode:
		Mode.PUSH_PULL:
			return ZMQ.SocketType.PUSH
		Mode.PUB_SUB:
			return ZMQ.SocketType.PUB
		Mode.REQ_REP:
			return ZMQ.SocketType.REQ

func get_zmq_in_connection_mode():
	match zmq_mode:
		Mode.PUSH_PULL:
			return ZMQ.ConnectionMode.CONNECT
		Mode.PUB_SUB:
			return ZMQ.ConnectionMode.CONNECT
		Mode.REQ_REP:
			return ZMQ.ConnectionMode.CONNECT

func get_zmq_out_connection_mode():
	match zmq_mode:
		Mode.PUSH_PULL:
			return ZMQ.ConnectionMode.BIND
		Mode.PUB_SUB:
			return ZMQ.ConnectionMode.BIND
		Mode.REQ_REP:
			return ZMQ.ConnectionMode.BIND

func get_zmq_receive_on_sender():
	match zmq_mode:
		Mode.PUSH_PULL:
			return false
		Mode.PUB_SUB:
			return false
		Mode.REQ_REP:
			return true

# Called when the node enters the scene tree for the first time.

func get_zmq_test_mode_string():
	match zmq_mode:
		Mode.PUSH_PULL:
			return "PUSH_PULL"
		Mode.PUB_SUB:
			return "PUB_SUB"
		Mode.REQ_REP:
			return "REQ_REP"

func _ready():
	for argument in OS.get_cmdline_args():
		if argument == "--push_pull" || argument == "--push-pull":
			zmq_mode = Mode.PUSH_PULL
		elif argument == "--pub_sub" || argument == "--pub-sub":
			zmq_mode = Mode.PUB_SUB
		elif argument == "--req_rep" || argument == "--req-rep":
			zmq_mode = Mode.REQ_REP

	zmq_receiver = create_zmq_receiver()

	add_child(zmq_receiver)

	print("====== ZMQ Test Mode: ", get_zmq_test_mode_string(), " ======")

	# Message input Handler NOTE: exclusive, one or the other
	#zmq_receiver.onMessageString(func(s: String):
		#print("[ZMQ Receiver] Received: ", s)
	#)
	
	zmq_receiver.onMessageBytes(func(bytes: PackedByteArray):
		print("[ZMQ Receiver] Received bytes: ", bytes.size())
		# 1️⃣  First part = topic  ▸ save it and return
		if pending_topic == "":
			pending_topic = _bytes_to_string(bytes)
			print("topic:", pending_topic)
			return

		# 2️⃣  Second part = payload  ▸ handle & then clear the latch
		var topic : String = pending_topic        # copy so we can clear safely
		pending_topic = ""                # ready for the next message

		match topic:
			"STRING_TOPIC":
				print(_bytes_to_string(bytes))

			"IMAGE_TOPIC":
				var img := _bytes_to_image(bytes)
				if img:
					# hop back to the main thread for the GPU upload
					call_deferred("_apply_image_as_texture", img)
				else:
					push_warning("Corrupt IMAGE_TOPIC payload")

			_:
				push_warning("Unsupported ZMQ topic: " + topic)
	)


func _apply_image_as_texture(img: Image) -> void:
	# This method runs on the main thread
	
	# Image lives in RAM – you can create, read, resize, or convert it safely from any thread.
	# ImageTexture.create_from_image() talks to Godot’s RenderingServer to upload those pixels to the GPU.
	#
	# The RenderingServer is strictly single-threaded.
	# If you call it from outside the main thread it refuses the request and silently returns a dummy 1 × 1 white texture (so the engine doesn’t crash). That’s the blank square you saw.
	#
	# The ZeroMQ plugin runs its I/O callback on its own worker thread, so every line inside _on_zmq_bytes() executed off the main thread.
	# 
	# By moving only the GPU-touching bit (create_from_image and sprite.texture = …) into a call_deferred() we make sure it runs on the main thread, where the RenderingServer is happy to accept it. The PNG → Image decode stays in the worker thread because it’s pure CPU work.
	
	var tex := ImageTexture.create_from_image(img)
	var sprite := get_parent() as Sprite2D
	sprite.texture  = tex
	sprite.centered = true
	sprite.scale = Vector2(1.0,1.0)
	sprite.position = DisplayServer.window_get_size() * 0.5

# -----------------------------------------------------------------
# Convert a PackedByteArray -> String  (assumes UTF-8, ZMQ’s default)
# -----------------------------------------------------------------
func _bytes_to_string(bytes: PackedByteArray) -> String:
	# Godot 4: PackedByteArray.get_string_from_utf8()
	# (falls back to replacement chars on invalid sequences)
	return bytes.get_string_from_utf8()

func _bytes_to_image(bytes: PackedByteArray) -> Image:
	var img := Image.new()
	var err := img.load_png_from_buffer(bytes)
	if err != OK:
		return null

	# guarantee renderable format
	if img.get_format() != Image.FORMAT_RGBA8:
		img.convert(Image.FORMAT_RGBA8)

	# resize + pad to 1080×1080 (optional, same logic as before)
	var scale:float = float(target_size.x) / max(img.get_width(), img.get_height())
	if scale != 1.0:
		prints("rescaling to", scale)
		var new_size := Vector2i(img.get_width() * scale, img.get_height() * scale) 
		img.resize(new_size.x, new_size.y, Image.INTERPOLATE_LANCZOS)

		var canvas := Image.create(target_size.x, target_size.y, false, Image.FORMAT_RGBA8)
		canvas.fill(Color(0,0,0,0))
		canvas.blit_rect(img, Rect2i(Vector2i.ZERO, new_size), (target_size - new_size)/2)
	
		return canvas
	return img
	
func _bytes_to_texture(bytes: PackedByteArray) -> Texture2D:
	var img := Image.new()
	
	# try PNG first, then JPEG, then (optionally) WebP
	var err := OK
	match image_format:
		"png", "PNG":
			print("Trying to load png")
			err = img.load_png_from_buffer(bytes)
		"jpg", "jpeg", "JPG", "JPEG":
			err = img.load_jpg_from_buffer(bytes)
		"webp", "WEBP":
			err = img.load_webp_from_buffer(bytes)   # drop this if you don’t need WebP

	if err != OK:
		prints("Invalid image received")
		return null       # couldn’t decode
	
	prints("Image format:", img.get_format(), "size:", img.get_size())
	
	# **** 1. guarantee a renderable format ****
	if img.get_format() != Image.FORMAT_RGB8 and img.get_format() != Image.FORMAT_RGBA8:
		print("Converting to RGB8")
		img.convert(Image.FORMAT_RGB8)

	# **** 2. (optional) save to disk once to sanity-check ****
	# comment this out after you verify the image really is noise
	#img.save_png("user://debug.png")

	# resize to fit TARGET while preserving aspect
	#var scale : float = float(target_size.x) / max(img.get_width(), img.get_height())
	#var new_size := Vector2i(img.get_width() * scale, img.get_height() * scale)
	#img.resize(new_size.x, new_size.y, Image.INTERPOLATE_LANCZOS)

	# composite onto 1080×1080 transparent canvas
	#var canvas := Image.create(target_size.x, target_size.y, false, img.get_format())
	#canvas.fill(Color(0,0,0,0))
	#canvas.blit_rect(img, Rect2i(Vector2i.ZERO, new_size), (target_size - new_size) / 2)

	return ImageTexture.create_from_image(img)

func _exit_tree():
	zmq_receiver.stop()
	remove_child(zmq_receiver)




#func _ready():
	#
	#self.onMessageString(func(s: String):
		#print("[ZMQ Receiver] Received: ", s)
	#)
	#self.onMessageBytes(func(bytes: PackedByteArray):
		#print("[ZMQ Receiver] Received: ", bytes.size())
	#)
#
#func _exit_tree():
	#self.stop()
