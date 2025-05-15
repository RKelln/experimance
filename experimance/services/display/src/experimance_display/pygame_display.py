import pygame

import cv2
import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GL import shaders

# returns screen, width, height
def get_fullscreen_display(display_index) -> tuple:
    display_index = min(display_index, pygame.display.get_num_displays() - 1)
    # Get display information
    # FIXME: This is not working as expected
    # display_info = pygame.display.Info()
    # print(display_info)
    # width = display_info.current_w
    # height = display_info.current_h

    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=display_index)
    width, height = screen.get_size()
    return screen, width, height

class FullscreenImageDisplay:
    def __init__(self, 
                 width=1920, 
                 height=1080, 
                 window_title="Experimance", 
                 do_fullscreen=False,
                 display_index=0):
        pygame.init()
        self.screen = None
        self.image = None
        self.running = False
        self._width = width
        self._height = height
        self.fullscreen = do_fullscreen
        self.window_title = window_title
        self.display_index = display_index


    def start(self):
        print("Starting pygame display")

        # Set up the display for fullscreen on the second monitor
        if self.fullscreen:
            self.screen, self.width, self.height = get_fullscreen_display(self.display_index)
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption(self.window_title)

        self.running = True


    def stop(self):
        self.running = False
        pygame.quit()


    # image is numpy array
    def render(self, image):
        if not self.running:
            return
        if image is None:
            print("pygame render(): Image is None")
            return

        if type(image) == np.ndarray:
            pass
        elif type(image) == Image.Image:
            image = np.array(image)
        else:
            raise Exception('render function received input of unknown type')

        # Ensure the image is in RGB format (Pygame expects RGB)
        if image.shape[2] == 4:
            # Image has alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            # Image is BGR, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a Pygame surface without copying data
        image_surface = pygame.image.frombuffer(image.tobytes(), image.shape[1::-1], 'RGB')

        # Scale the surface if necessary (hardware accelerated)
        if (image.shape[1] != self._width) or (image.shape[0] != self._height):
            image_surface = pygame.transform.smoothscale(image_surface, (self._width, self._height))

        # image_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))

        # # scale the image to fit the screen resolution
        # image_surface = pygame.transform.scale(image_surface, (self._width, self._height))

        # # clamp and set correct type
        # image = np.clip(image, 0, 255).astype(np.uint8)

        # # reshape if there is a mismatched between supply image size and window size
        # if image.shape[1] != self.width or image.shape[0] != self.height:
        #     image = cv2.resize(image, (self.width, self.height))

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = np.rot90(image)  # Pygame requires rotation to match orientation

        # surf = pygame.surfarray.make_surface(image)

        self.screen.blit(image_surface, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop()


class OpenGLImageDisplay:
    def __init__(self, width=1920, height=1080, window_title="Experimance", do_fullscreen=False, display_index=0):
        pygame.init()
        self.running = True
        self.window_title = window_title
        self.do_fullscreen = do_fullscreen
        self.display_index = display_index

        # Set display flags
        display_flags = pygame.OPENGL | pygame.DOUBLEBUF
        if self.do_fullscreen:
            display_flags |= pygame.FULLSCREEN
            # Get the number of displays
            num_displays = pygame.display.get_num_displays()
            if self.display_index >= num_displays:
                print(f"Display index {self.display_index} is out of range, using display 0.")
                self.display_index = 0

            # Get the size of the specified display
            desktop_sizes = pygame.display.get_desktop_sizes()
            self.width, self.height = desktop_sizes[self.display_index]

            # Initialize the display with the specified display index
            pygame.display.set_mode((self.width, self.height), display_flags, display=self.display_index)
        else:
            self.width = width
            self.height = height
            # Initialize the display
            pygame.display.set_mode((self.width, self.height), display_flags)

        pygame.display.set_caption(self.window_title)

        # Set up OpenGL
        self.init_opengl()

        # Initialize texture variables
        self.texture_id = None
        self.texture_initialized = False
        self.texture_width = None
        self.texture_height = None

    def init_opengl(self):
        # Set the viewport to cover the new window
        glViewport(0, 0, self.width, self.height)

        # Set projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)

        # Set modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def start(self):
        print("Starting OpenGL display")

    def stop(self):
        self.running = False
        pygame.quit()

    def initialize_texture(self, image_width, image_height):
        if self.texture_id is not None:
            glDeleteTextures([self.texture_id])
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        # Set texture parameters for scaling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Allocate texture storage (initialize texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_width, image_height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

        # Update texture dimensions
        self.texture_width = image_width
        self.texture_height = image_height

        self.texture_initialized = True

    def render(self, image):
        if not self.running:
            return
        if image is None:
            print("pygame_display render(): Image is None")
            return
        
        image_height, image_width = image.shape[0], image.shape[1]

        # Check if texture needs to be initialized or re-initialized
        if not self.texture_initialized or \
                (image_width != self.texture_width or image_height != self.texture_height):
            self.initialize_texture(image_width, image_height)

        # Flip image vertically (OpenGL's coordinate system)
        image = cv2.flip(image, 0)

        # Ensure image is in RGB format
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Image must have 3 (RGB) or 4 (RGBA) channels")

        # Bind the texture
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        # Update the texture data
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGB, GL_UNSIGNED_BYTE, image)

        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Enable textures
        glEnable(GL_TEXTURE_2D)

        # Calculate aspect ratios
        image_aspect = image_width / image_height
        screen_aspect = self.width / self.height

        # Determine the scaling factors to maintain aspect ratio
        if image_aspect > screen_aspect:
            # Image is wider than screen aspect ratio
            scale_width = self.width
            scale_height = self.width / image_aspect
        else:
            # Image is taller than screen aspect ratio
            scale_height = self.height
            scale_width = self.height * image_aspect

        # Calculate offsets to center the image
        offset_x = (self.width - scale_width) / 2
        offset_y = (self.height - scale_height) / 2

        # Draw the textured quad
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(offset_x, offset_y)
        glTexCoord2f(1, 0)
        glVertex2f(offset_x + scale_width, offset_y)
        glTexCoord2f(1, 1)
        glVertex2f(offset_x + scale_width, offset_y + scale_height)
        glTexCoord2f(0, 1)
        glVertex2f(offset_x, offset_y + scale_height)
        glEnd()

        # Swap buffers
        pygame.display.flip()

        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.stop()

    @property
    def screen_size(self):
        return self.width, self.height



class OpenGLImageShaderDisplay:
    def __init__(self, width=1920, height=1080, window_title="Experimance", do_fullscreen=False, 
                 display_index=0, show_fps=False, circular_mask=False):
        pygame.init()
        pygame.font.init()  # Initialize the font module
        self.running = True
        self.window_title = window_title
        self.do_fullscreen = do_fullscreen
        self.display_index = display_index
        self.show_fps = show_fps  # Option to show FPS
        self.clock = pygame.time.Clock()
        self.circular_mask = circular_mask

        # Set display flags
        display_flags = pygame.OPENGL | pygame.DOUBLEBUF
        if self.do_fullscreen:
            display_flags |= pygame.FULLSCREEN
            # Get the number of displays
            num_displays = pygame.display.get_num_displays()
            if self.display_index >= num_displays:
                print(f"Display index {self.display_index} is out of range, using display 0.")
                self.display_index = 0

            # Get the size of the specified display
            desktop_sizes = pygame.display.get_desktop_sizes()
            self.width, self.height = desktop_sizes[self.display_index]

            # Initialize the display with the specified display index
            pygame.display.set_mode((self.width, self.height), display_flags, display=self.display_index)
        else:
            self.width = width
            self.height = height
            # Initialize the display
            pygame.display.set_mode((self.width, self.height), display_flags)

        pygame.display.set_caption(self.window_title)

        # Set up OpenGL
        self.init_opengl()

        # Initialize texture and PBOs
        self.texture_id = glGenTextures(1)
        self.pbo_ids = glGenBuffers(2)  # Double-buffering
        self.next_pbo = 0
        self.texture_width = None
        self.texture_height = None

        # Compile shaders
        self.compile_shaders()

        # Initialize quad VBO
        self.quad_vbo = glGenBuffers(1)
        self.previous_aspect_ratio = None  # To track aspect ratio changes

        # Initialize FPS display
        if self.show_fps:
            self.init_fps_display()

    def init_opengl(self):
        # Set the viewport to cover the new window
        glViewport(0, 0, self.width, self.height)
        glEnable(GL_TEXTURE_2D)

        # Clear color
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def compile_shaders(self):
        vertex_shader_source = """
        #version 120
        attribute vec2 position;
        attribute vec2 texcoord;
        varying vec2 v_texcoord;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            v_texcoord = vec2(texcoord.x, 1.0 - texcoord.y); // Flip the texture coordinate here
        }
        """

        fragment_shader_source = """
        #version 120
        uniform sampler2D tex;
        uniform bool u_circular_mask;
        varying vec2 v_texcoord;
        void main() {
            vec4 color = texture2D(tex, v_texcoord);
            if (u_circular_mask) {
                vec2 center = vec2(0.5, 0.5);
                float radius = 0.5;
                float dist = distance(v_texcoord, center);
                if (dist > radius) {
                    color = vec4(0.0, 0.0, 0.0, 1.0);
                }
            }
            gl_FragColor = color;
        }
        """

        # Compile shaders
        self.shader_program = shaders.compileProgram(
            shaders.compileShader(vertex_shader_source, GL_VERTEX_SHADER),
            shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
        )

        # Get attribute and uniform locations
        self.position_loc = glGetAttribLocation(self.shader_program, 'position')
        self.texcoord_loc = glGetAttribLocation(self.shader_program, 'texcoord')
        self.texture_uniform = glGetUniformLocation(self.shader_program, 'tex')
        self.circular_mask_uniform = glGetUniformLocation(self.shader_program, 'u_circular_mask')


    def init_fps_display(self):
        # Font settings
        self.font_size = 24
        self.font_color = (255, 255, 255)  # White color
        self.font = pygame.font.SysFont("Arial", self.font_size)
        self.fps_texture_id = glGenTextures(1)
        self.fps_position = (10, 10)  # Position on screen

    def start(self):
        print("Starting OpenGL display")

    def stop(self):
        self.running = False
        pygame.quit()

    def initialize_texture(self, image_width, image_height):
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        # Set texture parameters for scaling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # Allocate texture storage (initialize texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_width, image_height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

        # Initialize PBOs
        for pbo_id in self.pbo_ids:
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, image_width * image_height * 3, None, GL_STREAM_DRAW)

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        # Update texture dimensions
        self.texture_width = image_width
        self.texture_height = image_height

    def update_quad_vbo(self, image_width, image_height):
        # Calculate aspect ratios
        image_aspect = image_width / image_height
        screen_aspect = self.width / self.height

        if image_aspect == self.previous_aspect_ratio:
            return  # No need to update VBO

        self.previous_aspect_ratio = image_aspect

        if image_aspect > screen_aspect:
            # Image is wider than screen
            scale_x = 1.0
            scale_y = screen_aspect / image_aspect
        else:
            # Image is narrower than screen
            scale_x = image_aspect / screen_aspect
            scale_y = 1.0

        # Define quad vertices with adjusted positions
        quad_vertices = np.array([
            -scale_x, -scale_y,  0.0, 0.0,
             scale_x, -scale_y,  1.0, 0.0,
             scale_x,  scale_y,  1.0, 1.0,
            -scale_x,  scale_y,  0.0, 1.0,
        ], dtype=np.float32)

        # Update the quad VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render_fps(self, fps=None):
        # Render the FPS text to a Pygame surface
        if fps is None:
            fps = self.clock.get_fps()
        fps_text = f"FPS: {fps:.0f}"
        text_surface = self.font.render(fps_text, True, self.font_color)
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        text_width, text_height = text_surface.get_size()

        # Create an OpenGL texture from the text surface
        glBindTexture(GL_TEXTURE_2D, self.fps_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_width, text_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        # Calculate position in normalized coordinates
        x = -1 + (self.fps_position[0] / self.width) * 2
        y = 1 - (self.fps_position[1] / self.height) * 2 - (text_height / self.height) * 2

        # Define quad vertices for the text
        quad_vertices = np.array([
            x, y,  0.0, 1.0,
            x + (text_width / self.width) * 2, y,  1.0, 1.0,
            x + (text_width / self.width) * 2, y + (text_height / self.height) * 2,  1.0, 0.0,
            x, y + (text_height / self.height) * 2,  0.0, 0.0,
        ], dtype=np.float32)

        # Create VBO for text quad
        text_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, text_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)

        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Use shader program
        glUseProgram(self.shader_program)

        # Bind texture
        glBindTexture(GL_TEXTURE_2D, self.fps_texture_id)
        glUniform1i(self.texture_uniform, 0)  # Texture unit 0

        # Enable vertex attributes
        glEnableVertexAttribArray(self.position_loc)
        glEnableVertexAttribArray(self.texcoord_loc)

        # Set vertex attribute pointers
        stride = 4 * sizeof(GLfloat)
        glVertexAttribPointer(self.position_loc, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glVertexAttribPointer(self.texcoord_loc, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * sizeof(GLfloat)))

        # Draw the text quad
        glDrawArrays(GL_QUADS, 0, 4)

        # Disable vertex attributes
        glDisableVertexAttribArray(self.position_loc)
        glDisableVertexAttribArray(self.texcoord_loc)

        # Unbind VBO and texture
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Unuse shader program
        glUseProgram(0)

        # Disable blending
        glDisable(GL_BLEND)

        # Delete the text VBO
        glDeleteBuffers(1, [text_vbo])

    def render(self, image, fps=None):
        if not self.running:
            return

        image_height, image_width = image.shape[0], image.shape[1]

        # Check if texture needs to be initialized or re-initialized
        if self.texture_width != image_width or self.texture_height != image_height:
            self.initialize_texture(image_width, image_height)
            self.previous_aspect_ratio = None  # Force VBO update on size change

        # Update quad VBO if aspect ratio changes
        self.update_quad_vbo(image_width, image_height)

        # Ensure image is of a supported data type
        if image.dtype != np.uint8:
            if image.dtype == np.float64 or image.dtype == np.float32:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                raise ValueError(f"Unsupported image data type: {image.dtype}")

        # Ensure image is in RGB format
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Image must have 3 (RGB) or 4 (RGBA) channels")

        # Bind the PBO for the next frame
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo_ids[self.next_pbo])

        # Map the buffer object into client's memory
        ptr = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY)
        if ptr:
            # Copy image data into the mapped buffer
            from ctypes import memmove
            memmove(ptr, image.ctypes.data, image_width * image_height * 3)
            glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER)
        else:
            print("Failed to map PBO")

        # Bind the texture
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        # Update texture with PBO
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGB, GL_UNSIGNED_BYTE, None)

        # Unbind PBO
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Use shader program
        glUseProgram(self.shader_program)

        # Set texture uniform
        glUniform1i(self.texture_uniform, 0)  # Texture unit 0

        # Set circular mask uniform
        glUniform1i(self.circular_mask_uniform, int(self.circular_mask))

        # Bind VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)

        # Enable vertex attributes
        glEnableVertexAttribArray(self.position_loc)
        glEnableVertexAttribArray(self.texcoord_loc)

        # Set vertex attribute pointers
        stride = 4 * sizeof(GLfloat)
        glVertexAttribPointer(self.position_loc, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glVertexAttribPointer(self.texcoord_loc, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * sizeof(GLfloat)))

        # Bind image texture
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        # Draw the quad
        glDrawArrays(GL_QUADS, 0, 4)

        # Disable vertex attributes
        glDisableVertexAttribArray(self.position_loc)
        glDisableVertexAttribArray(self.texcoord_loc)

        # Unbind VBO and texture
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Unuse shader program
        glUseProgram(0)

        # Render FPS if enabled
        if self.show_fps:
            self.render_fps(fps)

        # Swap buffers
        pygame.display.flip()

        # Process events
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         self.stop()
        #     elif event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_ESCAPE:
        #             self.stop()

        # Swap PBO indices for next frame
        self.next_pbo = (self.next_pbo + 1) % 2

        self.clock.tick()

    @property
    def screen_size(self):
        return self.width, self.height


if __name__ == "__main__":
    import cv2
    import time

    #display = FullscreenImageDisplay(do_fullscreen=True)
    display = OpenGLImageShaderDisplay(do_fullscreen=True, display_index=1, show_fps=True, circular_mask=True)
    display.start()

    # Simulate image updates (replace with your image source)
    # For testing, we'll generate a random image each frame
    clock = pygame.time.Clock()

    image = cv2.imread("images/test.png")
    assert image is not None, "Image not found"
    # test fps: 30
    while display.running:
        t = time.monotonic()
        # Simulate an image (replace with actual image acquisition)
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        fps = clock.get_fps()
        display.render(image)
        duration = time.monotonic() - t
        print(f"Render time: {duration:.4f}, FPS: {fps:.2f}", end='\r')
        clock.tick()

    display.stop()