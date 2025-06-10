OK now that the basic pyglet_test is working and showing decent frame rates (45fps avg) let's create services/display/src/diplay.py that implements the display service.

This service displays the latest generated satellite landscape image, and transitions smoothly to the next. It also displays a masked video that displays only when the user interacts with the sand, the mask denotes the area of sand that changed from their manipulation. Experimance core knows the state of the sand and controls the image generation requests and knows when the requests are fulfilled. The flow looks like:

User manipulates the sand, depth camera reads the changes, sends the difference in depth to core which processes the iamge to create a mask the denotes which areas have changed, this gets sent to display, which then updates its mask so tha the video appears only where the user has changed the sand. In parallel a image generation request is made by core and when the image is generated it is sent to display. Optionally the image and previous image are sent to a transition creator that generates a smooth transition and the transition is sent along with the new image to display. It is possible that new images are also generated without user intervention after a certain time period.

Display service should accept images or filepaths over zmq:
1) satellite landscape images: it will crossfade to the latest image on receive
2) mask images that control the mask of the video play back, the mask should fade in quickly and then fade out when a new satellite landscape image fades in
Note: I think it is fine to accept an image path in both cases and then load that. We want fast and responsive, but if core and display are not on the same computer then we'd need to send images. These could be different topics/types of messages.

It should also accept transition videos paths over zmq. These would replace the default crossfade between images, so perhaps the next image message also includes a transition video path? Sending video over zmq seems bad but may be needed.

Display service should also accept text to display:
3) Each text should have:
- an id that refers to it
- a speaker attribute that could change how it is displayed
- a duration for it to display and then fade out quickly or infnite  
- allow for a remove text event (by id) also sent by zmq

Images, text and other signals will come from experimance core, so it can subscribe to a bunch of topics most likely.

Read the revelant docs, check image_service as an example, then ask any follow-up questions, then let's write up a design doc, a TODO list before starting any coding. As always we want clean, elegant and readable code using best practices.