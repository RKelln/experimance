#!/usr/bin/env python3
"""
Simple image viewer for a folder tree with filtering by subdirectory name (era/biome),
ordered by filename (basename), supporting autoplay (fixed rate) or manual navigation via keys.

Usage examples:
  uv run scripts/view_images.py
  uv run scripts/view_images.py media/images/generated --era modern --biome tundra --delay 2.5

Controls when running:
  Right / n / Space: next image (space also toggles autoplay)
  Left / p: previous image
  Hold Right/Left/n/p: after 0.5s delay, auto-advance at 10fps
  a: toggle autoplay
  d: delete current image file (moves to trash/deletes permanently)
  q / Escape: quit
  + / - : increase / decrease delay (when autoplay)
  f: toggle fullscreen

Requirements:
  - Python 3.8+
  - pyglet

"""

import argparse
import sys
import os
from pathlib import Path
import time
from typing import List, Optional

try:
    import pyglet
    from pyglet import window, image as pyglet_image, clock, text
    from pyglet.window import key
except ImportError:
    print("Error: pyglet not found. Install with: uv add pyglet")
    sys.exit(1)

# Fix XCB threading issues when run via uv/packaging tools
import os
os.environ.setdefault('QT_X11_NO_MITSHM', '1')
os.environ.setdefault('MPLBACKEND', 'TkAgg')

# Import ctypes to set X11 threading before tkinter loads
try:
    import ctypes
    import ctypes.util
    # Load X11 library and initialize threading
    x11 = ctypes.cdll.LoadLibrary(ctypes.util.find_library('X11') or 'libX11.so.6')
    if hasattr(x11, 'XInitThreads'):
        x11.XInitThreads()
except Exception:
    pass

import argparse
import sys
import os
from pathlib import Path
import time
from typing import List

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}


def find_images(root: Path, era: str | None, biome: str | None) -> List[Path]:
    """Recursively collect image files under root, filter by era/biome substrings in path parts,
    and return a list sorted by filename (basename).
    """
    if not root.exists():
        return []

    files: List[Path] = []
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            parts = [part.lower() for part in p.relative_to(root).parts]
            ok = True
            if era:
                if not any(era.lower() in part for part in parts):
                    ok = False
            if biome:
                if not any(biome.lower() in part for part in parts):
                    ok = False
            if ok:
                files.append(p)

    # Sort by basename (filename only) then by full path as tiebreaker
    files.sort(key=lambda p: (p.name.lower(), str(p)))
    return files


class ImageViewer(window.Window):
    def __init__(self, image_paths: List[Path], delay: float = 3.0):
        if not image_paths:
            raise ValueError('No images to display.')

        self.paths = image_paths
        self.delay = max(0.1, float(delay))
        self.index = 0
        self.paused = True
        self.current_sprite: Optional[pyglet.sprite.Sprite] = None
        self.status_label: Optional[text.Label] = None

        # Key repeat for navigation with startup delay
        self.key_repeat_startup_delay = 0.3  # Wait 500ms before starting repeats
        self.key_repeat_speed = 0.1  # 100ms between repeats once started (10 fps)
        self.keys_held = {}  # Track key press time: {key: press_time}
        self.last_key_repeat = 0

        # Get screen size using pyglet display API (same as display service)
        try:
            display = pyglet.display.get_display()
            screens = display.get_screens()
            if screens:
                screen = screens[0]  # Use primary screen
                width = min(1024, screen.width)
                height = min(1024, screen.height)
            else:
                width, height = 1024, 1024
        except Exception:
            width, height = 1024, 1024
            
        super().__init__(
            width=width,
            height=height,
            caption='Image Viewer',
            resizable=True
        )
        
        # Center the window on screen if possible
        try:
            if 'screen' in locals():
                self.set_location((screen.width - width) // 2, (screen.height - height) // 2)
            else:
                self.set_location(100, 100)
        except Exception:
            pass
        
        # Create status label
        self.status_label = text.Label(
            '', font_name='Arial', font_size=12,
            x=10, y=10, color=(255, 255, 255, 255)
        )
        
        # Schedule autoplay updates
        clock.schedule_interval(self._update, 0.1)
        self._next_time = time.time() + self.delay
        
        # Load first image
        self.show_image(0)

    def on_draw(self):
        self.clear()
        if self.current_sprite:
            self.current_sprite.draw()
        if self.status_label:
            self.status_label.draw()

    def on_key_press(self, symbol, modifiers):
        # Track navigation keys for repeat functionality with press time
        if symbol in (key.RIGHT, key.LEFT, key.N, key.P):
            if symbol not in self.keys_held:  # Only record initial press time
                self.keys_held[symbol] = time.time()
            
        if symbol == key.RIGHT or symbol == key.N:
            self.next_image()
        elif symbol == key.LEFT or symbol == key.P:
            self.prev_image()
        elif symbol == key.SPACE or symbol == key.A:
            self.toggle_play()
        elif symbol == key.Q or symbol == key.ESCAPE:
            self.close()
        elif symbol == key.PLUS or symbol == key.EQUAL or symbol == key.COMMA:
            self.change_delay(-0.5)
        elif symbol == key.MINUS or symbol == key.PERIOD:
            self.change_delay(0.5)
        elif symbol == key.F:
            self.set_fullscreen(not self.fullscreen)
        elif symbol == key.D:
            self.delete_current_image()

    def on_key_release(self, symbol, modifiers):
        # Remove key from held dict when released
        self.keys_held.pop(symbol, None)

    def on_resize(self, width, height):
        super().on_resize(width, height)
        if self.status_label:
            self.status_label.y = 10
        # Reload current image to fit new window size
        self.show_image(self.index, resize=True)

    def show_image(self, idx: int, resize: bool = False):
        idx = idx % len(self.paths)
        path = self.paths[idx]
        
        try:
            # Load image with pyglet
            img = pyglet_image.load(str(path))
            
            # Scale to fit window while maintaining aspect ratio
            img_ratio = img.width / img.height
            win_ratio = self.width / self.height
            
            if img_ratio > win_ratio:
                # Image is wider, fit to width
                scale = self.width / img.width
            else:
                # Image is taller, fit to height
                scale = self.height / img.height
            
            # Don't upscale small images too much
            if scale > 1.0:
                scale = min(scale, 2.0)
            
            # Create sprite and position it centered
            self.current_sprite = pyglet.sprite.Sprite(img)
            self.current_sprite.scale = scale
            self.current_sprite.x = (self.width - img.width * scale) / 2
            self.current_sprite.y = (self.height - img.height * scale) / 2
            
            self.index = idx
            self._update_status()
            
        except Exception as e:
            print(f'Failed to load {path}: {e}', file=sys.stderr)

    def _update_status(self):
        if self.status_label:
            path = self.paths[self.index]
            status_text = f'[{self.index+1}/{len(self.paths)}] {path.name} | delay={self.delay:.1f}s | paused={self.paused}'
            self.status_label.text = status_text

    def next_image(self):
        self.show_image((self.index + 1) % len(self.paths))
        self._next_time = time.time() + self.delay

    def prev_image(self):
        self.show_image((self.index - 1) % len(self.paths))
        self._next_time = time.time() + self.delay

    def toggle_play(self):
        self.paused = not self.paused
        if not self.paused:
            self._next_time = time.time() + self.delay
        self._update_status()

    def change_delay(self, delta: float):
        if self.delay <= 0.5:
            if self.delay <= 0.1:
                self.delay = max(0.01, self.delay + delta*0.02)
            else:
                self.delay = max(0.1, self.delay + delta*0.2)
        else:
            self.delay = max(0.5, self.delay + delta)
        self._update_status()

    def delete_current_image(self):
        """Delete the current image file and remove it from the list."""
        if not self.paths:
            return
            
        current_path = self.paths[self.index]
        try:
            # Try to move to trash first (safer), fall back to permanent deletion
            try:
                import send2trash
                send2trash.send2trash(str(current_path))
                print(f"Moved to trash: {current_path.name}")
            except ImportError:
                # Fallback to permanent deletion if send2trash not available
                current_path.unlink()
                print(f"Deleted: {current_path.name}")
            
            # Remove from paths list
            self.paths.pop(self.index)
            
            # Handle empty list
            if not self.paths:
                print("No more images to display")
                self.close()
                return
                
            # Adjust index if we were at the end
            if self.index >= len(self.paths):
                self.index = len(self.paths) - 1
                
            # Show next/current image
            self.show_image(self.index)
            
        except Exception as e:
            print(f"Failed to delete {current_path.name}: {e}")
            self._update_status()

    def _update(self, dt):
        current_time = time.time()
        
        # Handle normal autoplay
        if not self.paused and current_time >= self._next_time:
            self.next_image()
        
        # Handle key repeat for navigation with startup delay
        if self.keys_held and current_time - self.last_key_repeat >= self.key_repeat_speed:
            for held_key, press_time in self.keys_held.items():
                # Only start repeating after startup delay has passed
                if current_time - press_time >= self.key_repeat_startup_delay:
                    if held_key in (key.RIGHT, key.N):
                        self.next_image()
                        self.last_key_repeat = current_time
                        break  # Only handle one key per update cycle
                    elif held_key in (key.LEFT, key.P):
                        self.prev_image()
                        self.last_key_repeat = current_time
                        break  # Only handle one key per update cycle

    def run(self):
        pyglet.app.run()


def parse_args():
    p = argparse.ArgumentParser(description='View images in a folder tree with era/biome filtering.')
    p.add_argument('root', nargs='?', default='media/images/generated', help='Root directory to search (default: media/images/generated)')
    p.add_argument('--era', '-e', help='Filter by era substring (case-insensitive)')
    p.add_argument('--biome', '-b', help='Filter by biome substring (case-insensitive)')
    p.add_argument('--delay', '-d', type=float, default=3.0, help='Autoplay delay in seconds (default 3.0)')
    p.add_argument('--autoplay', action='store_true', help='Start in autoplay mode')
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)

    images = find_images(root, args.era, args.biome)
    if not images:
        print(f'No images found under {root} with filters era={args.era} biome={args.biome}')
        sys.exit(2)

    try:
        viewer = ImageViewer(images, delay=args.delay)
    except Exception as e:
        print(str(e), file=sys.stderr)
        print('\nPlease ensure you have pyglet installed. Run:')
        print('\n  uv add pyglet\n')
        sys.exit(3)

    if args.autoplay:
        viewer.paused = False

    viewer.run()


if __name__ == '__main__':
    main()
