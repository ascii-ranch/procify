import pygame
import psutil
import time
import random
import math
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple
import logging
import os
from scipy.io import wavfile
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame.locals as pgl

# Set up logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create sounds directory
sounds_dir = "sounds"
if not os.path.exists(sounds_dir):
    os.makedirs(sounds_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'procify_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProcessNode:
    def __init__(self, pid: int, name: str, pos: Tuple[float, float], creation_time: float, is_initial: bool = False):
        self.pid = pid
        self.name = name
        self.pos = list(pos)
        self.target_pos = list(pos)  # Only used for new processes
        self.velocity = [0, 0]
        self.creation_time = creation_time
        self.alpha = 255
        self.radius = 5
        self.is_new = not is_initial
        self.time_alive = 0
        self.is_terminating = False
        # Brighter colors for new processes
        if self.is_new:
            self.color = (
                random.randint(180, 255) / 255.0,
                random.randint(180, 255) / 255.0,
                random.randint(180, 255) / 255.0
            )
        else:
            self.color = (
                random.randint(50, 150) / 255.0,
                random.randint(50, 150) / 255.0,
                random.randint(50, 150) / 255.0
            )
        self.parent_pid = None
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.network_connections = []
        self.last_update = time.time()
        self.net_io_counters = None
        self.last_bytes_sent = 0
        self.last_bytes_recv = 0
        self.network_activity = 0.0
        self.text_offset = [10, -10]
        self.connection_endpoints = {}  # Store stable connection endpoints
        
        try:
            proc = psutil.Process(pid)
            self.parent_pid = proc.ppid()
            proc.cpu_percent()
            self.memory_percent = proc.memory_percent()
            self.network_connections = proc.connections()
            # Initialize stable connection endpoints
            for conn in self.network_connections:
                if conn.status == 'ESTABLISHED' and conn.raddr:
                    key = f"{conn.raddr.ip}:{conn.raddr.port}"
                    if key not in self.connection_endpoints:
                        angle = random.uniform(0, 2 * math.pi)
                        self.connection_endpoints[key] = angle
            try:
                self.net_io_counters = proc.io_counters()
                self.last_bytes_sent = self.net_io_counters.write_bytes
                self.last_bytes_recv = self.net_io_counters.read_bytes
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            logger.info(f"{'Initial' if is_initial else 'New'} process node created - PID: {pid}, Name: {name}, Parent PID: {self.parent_pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Could not get process info for {pid} ({name}): {str(e)}")

class ProcessVisualizer:
    def __init__(self):
        logger.info("Initializing ProcessVisualizer")
        try:
            pygame.init()
            if not pygame.display.get_init():
                raise RuntimeError("Could not initialize pygame display")
                
            # Initialize OpenGL attributes before creating the window
            pygame.display.gl_set_attribute(pgl.GL_MULTISAMPLEBUFFERS, 1)
            pygame.display.gl_set_attribute(pgl.GL_MULTISAMPLESAMPLES, 4)
            pygame.display.gl_set_attribute(pgl.GL_DOUBLEBUFFER, 1)
            pygame.display.gl_set_attribute(pgl.GL_DEPTH_SIZE, 24)
            
            try:
                pygame.mixer.init(44100, -16, 2, 1024)
            except pygame.error:
                logger.warning("Could not initialize sound mixer. Sound effects will be disabled.")
            
            # Get display information for all monitors
            pygame.display.init()
            displays = pygame.display.get_desktop_sizes()
            
            # Use second monitor if available, otherwise use primary
            if len(displays) > 1:
                self.monitor_index = 1
                monitor_x = sum(d[0] for d in displays[:1])  # X offset for second monitor
                self.width = min(1200, displays[1][0] - 100)
                self.height = min(800, displays[1][1] - 100)
                os.environ['SDL_VIDEO_WINDOW_POS'] = f"{monitor_x + 50},{50}"
            else:
                self.monitor_index = 0
                self.width = min(1200, displays[0][0] - 100)
                self.height = min(800, displays[0][1] - 100)
                os.environ['SDL_VIDEO_WINDOW_POS'] = "50,50"
            
            # Create window with minimum size constraints
            self.min_width = 800
            self.min_height = 600
            
            try:
                self.screen = pygame.display.set_mode(
                    (self.width, self.height),
                    pgl.OPENGL | pgl.DOUBLEBUF | pgl.RESIZABLE
                )
            except pygame.error as e:
                raise RuntimeError(f"Could not create OpenGL window: {str(e)}")
            
            pygame.display.set_caption("Procify - Process Visualization")
            
            # Initialize OpenGL
            try:
                self.update_viewport()
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glEnable(GL_LINE_SMOOTH)
                glEnable(GL_POINT_SMOOTH)
                glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
                glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
            except Exception as e:
                raise RuntimeError(f"Could not initialize OpenGL: {str(e)}")
            
            # Initialize other attributes
            self.window_drag = False
            self.drag_offset = (0, 0)
            self.titlebar_height = 30
            self.min_edge_length = 150
            self.repulsion_strength = 2000
            self.attraction_strength = 0.1
            self.max_edge_length = 300
            self.animation_speed = 0.2
            self.last_frame_time = time.time()
            self.delta_time = 0.016
            
            # Create text surface
            try:
                self.text_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                self.font = pygame.font.Font(None, 24)
                self.small_font = pygame.font.Font(None, 18)
            except pygame.error as e:
                raise RuntimeError(f"Could not initialize text rendering: {str(e)}")
            
            self.generate_sound_effects()
            
            self.clock = pygame.time.Clock()
            self.nodes = {}
            self.last_process_check = 0
            self.process_check_interval = 0.05
            self.process_update_index = 0
            self.processes_per_update = 10
            self.running = True
            self.total_processes_monitored = 0
            
            self.offset_x = 0
            self.offset_y = 0
            self.zoom = 1.0
            self.dragging = False
            self.last_mouse_pos = None
            self.new_process_highlight_duration = 5.0
            
            self.show_network = True
            self.show_stats = True
            self.initial_startup = True
            self.movement_speed = 0.1
            self.node_spacing = 100
            
            logger.info(f"ProcessVisualizer initialized with window size: {self.width}x{self.height}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ProcessVisualizer: {str(e)}")
            raise

    def generate_sound_effects(self):
        """Generate ascending and descending ping sounds"""
        sample_rate = 44100
        duration = 0.1  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Ascending ping (process start)
        start_freq = 440
        end_freq = 880
        start_sound = np.sin(2 * np.pi * np.linspace(start_freq, end_freq, len(t)) * t)
        # Convert to stereo by duplicating the mono channel and ensure C-contiguous
        start_sound = np.ascontiguousarray(np.vstack((start_sound, start_sound)).T)
        start_sound = np.int16(start_sound * 32767)
        wavfile.write(os.path.join(sounds_dir, "process_start.wav"), sample_rate, start_sound)
        
        # Descending ping (process end)
        end_sound = np.sin(2 * np.pi * np.linspace(end_freq, start_freq, len(t)) * t)
        # Convert to stereo by duplicating the mono channel and ensure C-contiguous
        end_sound = np.ascontiguousarray(np.vstack((end_sound, end_sound)).T)
        end_sound = np.int16(end_sound * 32767)
        wavfile.write(os.path.join(sounds_dir, "process_end.wav"), sample_rate, end_sound)
        
        # Load the sounds
        self.start_sound = pygame.mixer.Sound(os.path.join(sounds_dir, "process_start.wav"))
        self.end_sound = pygame.mixer.Sound(os.path.join(sounds_dir, "process_end.wav"))
        
        # Set volume
        self.start_sound.set_volume(0.3)
        self.end_sound.set_volume(0.3)

    def update_viewport(self):
        """Update OpenGL viewport after window resize or move"""
        try:
            glViewport(0, 0, self.width, self.height)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluOrtho2D(0, self.width, self.height, 0)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
        except Exception as e:
            logger.error(f"Error updating viewport: {str(e)}")

    def screen_to_world(self, screen_pos):
        """Convert screen coordinates to world coordinates"""
        x = (screen_pos[0] - self.width/2) / self.zoom - self.offset_x
        y = (screen_pos[1] - self.height/2) / self.zoom - self.offset_y
        return (x, y)

    def world_to_screen(self, world_pos):
        """Convert world coordinates to screen coordinates"""
        x = (world_pos[0] + self.offset_x) * self.zoom + self.width/2
        y = (world_pos[1] + self.offset_y) * self.zoom + self.height/2
        return (x, y)

    def get_spawn_position(self) -> Tuple[float, float]:
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(100, 300)
        x = math.cos(angle) * radius
        y = math.sin(angle) * radius
        return (x, y)

    def handle_input(self):
        current_time = time.time()
        self.delta_time = min(current_time - self.last_frame_time, 0.1)  # Cap at 100ms
        self.last_frame_time = current_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                logger.info("Received quit signal")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    logger.info("Received escape key - shutting down")
                elif event.key == pygame.K_n:
                    self.show_network = not self.show_network
                elif event.key == pygame.K_s:
                    self.show_stats = not self.show_stats
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    if mouse_pos[1] < self.titlebar_height:
                        self.window_drag = True
                        self.drag_offset = mouse_pos
                    else:
                        self.dragging = True
                        self.last_mouse_pos = event.pos
                elif event.button == 4:
                    self.zoom *= 1.1
                elif event.button == 5:
                    self.zoom /= 1.1
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.window_drag = False
                    self.dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.window_drag:
                    new_pos = pygame.mouse.get_pos()
                    dx = new_pos[0] - self.drag_offset[0]
                    dy = new_pos[1] - self.drag_offset[1]
                    x, y = pygame.display.get_window_position()
                    try:
                        pygame.display.set_mode((self.width, self.height), 
                                              pgl.OPENGL | pgl.DOUBLEBUF | pgl.RESIZABLE)
                        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x + dx},{y + dy}"
                        self.update_viewport()
                    except Exception as e:
                        logger.error(f"Error during window drag: {str(e)}")
                    self.drag_offset = new_pos
                elif self.dragging:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    self.offset_x += dx / self.zoom
                    self.offset_y += dy / self.zoom
                    self.last_mouse_pos = event.pos
            elif event.type == pygame.VIDEORESIZE:
                # Enforce minimum window size
                self.width = max(event.w, self.min_width)
                self.height = max(event.h, self.min_height)
                try:
                    self.screen = pygame.display.set_mode((self.width, self.height), 
                                                        pgl.OPENGL | pgl.DOUBLEBUF | pgl.RESIZABLE)
                    self.update_viewport()
                except Exception as e:
                    logger.error(f"Error during resize: {str(e)}")
                    # Revert to previous size if resize fails
                    self.width = event.w
                    self.height = event.h
                    self.screen = pygame.display.set_mode((self.width, self.height), 
                                                        pgl.OPENGL | pgl.DOUBLEBUF | pgl.RESIZABLE)
                    self.update_viewport()

    def draw_circle(self, x, y, radius, color):
        glColor4f(color[0], color[1], color[2], color[3] if len(color) > 3 else 1.0)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(x, y)
        segments = 32
        for i in range(segments + 1):
            angle = i * (2.0 * math.pi / segments)
            glVertex2f(x + math.cos(angle) * radius,
                      y + math.sin(angle) * radius)
        glEnd()

    def draw_line(self, start_pos, end_pos, color, width=1):
        glLineWidth(width)
        glColor4f(color[0], color[1], color[2], color[3] if len(color) > 3 else 1.0)
        glBegin(GL_LINES)
        glVertex2f(start_pos[0], start_pos[1])
        glVertex2f(end_pos[0], end_pos[1])
        glEnd()

    def apply_layout_forces(self):
        """Apply force-directed layout to reduce clustering"""
        if not self.nodes:
            return
            
        forces = {pid: [0, 0] for pid in self.nodes}
        
        # Apply repulsion between all nodes
        for pid1, node1 in self.nodes.items():
            for pid2, node2 in self.nodes.items():
                if pid1 != pid2:
                    dx = node1.pos[0] - node2.pos[0]
                    dy = node1.pos[1] - node2.pos[1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance < 0.1:
                        distance = 0.1
                    
                    force = self.repulsion_strength / (distance * distance)
                    forces[pid1][0] += force * dx / distance
                    forces[pid1][1] += force * dy / distance
        
        # Apply attraction for connected nodes
        for node in self.nodes.values():
            if node.parent_pid in self.nodes:
                parent = self.nodes[node.parent_pid]
                dx = parent.pos[0] - node.pos[0]
                dy = parent.pos[1] - node.pos[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > self.min_edge_length:
                    force = self.attraction_strength * (distance - self.min_edge_length)
                    forces[node.pid][0] += force * dx / distance
                    forces[node.pid][1] += force * dy / distance
        
        # Apply forces smoothly using delta time
        for pid, node in self.nodes.items():
            if not node.is_new:
                force_x = forces[pid][0]
                force_y = forces[pid][1]
                
                # Limit maximum force
                force_magnitude = math.sqrt(force_x*force_x + force_y*force_y)
                if force_magnitude > 5:
                    force_x *= 5/force_magnitude
                    force_y *= 5/force_magnitude
                
                # Apply smooth movement using delta time
                node.pos[0] += force_x * self.animation_speed * self.delta_time * 60
                node.pos[1] += force_y * self.animation_speed * self.delta_time * 60

    def draw(self):
        # Apply layout forces before drawing
        self.apply_layout_forces()
        
        # Clear both OpenGL and text surface
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self.text_surface.fill((0, 0, 0, 0))
        
        # Draw title bar
        glColor4f(0.2, 0.2, 0.2, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(self.width, 0)
        glVertex2f(self.width, self.titlebar_height)
        glVertex2f(0, self.titlebar_height)
        glEnd()
        
        try:
            # Create a new surface each frame to avoid memory issues
            self.text_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            self.text_surface.fill((0, 0, 0, 0))
            
            title_text = self.font.render("Procify - Process Visualization (Drag to move)", True, (200, 200, 200))
            self.text_surface.blit(title_text, (10, 5))
            
            # After all OpenGL drawing, render the text surface
            text_data = pygame.image.tostring(self.text_surface, 'RGBA', True)
            glRasterPos2d(0, self.height)
            glPixelZoom(1, -1)
            glDrawPixels(self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            
        except Exception as e:
            logger.error(f"Error rendering text: {str(e)}")
            # Continue without text rendering if there's an error
            pass

        # First, identify processes that should be visible
        visible_processes = set()
        for node in self.nodes.values():
            if (node.is_new or 
                node.is_terminating or
                node.alpha < 255 or 
                any(conn.status == 'ESTABLISHED' for conn in node.network_connections)):
                visible_processes.add(node.pid)
                # Also add parent if it exists
                if node.parent_pid in self.nodes:
                    visible_processes.add(node.parent_pid)

        # Draw network connections if enabled
        if self.show_network:
            for node in self.nodes.values():
                if node.pid in visible_processes:
                    screen_pos = self.world_to_screen(node.pos)
                    
                    # Draw established connections using stored endpoints
                    for conn in node.network_connections:
                        if conn.status == 'ESTABLISHED' and conn.raddr:
                            key = f"{conn.raddr.ip}:{conn.raddr.port}"
                            
                            # Create stable endpoint if it doesn't exist
                            if key not in node.connection_endpoints:
                                angle = random.uniform(0, 2 * math.pi)
                                node.connection_endpoints[key] = angle
                            
                            # Use stored angle for stable endpoint position
                            angle = node.connection_endpoints[key]
                            end_pos = (
                                screen_pos[0] + math.cos(angle) * 50,
                                screen_pos[1] + math.sin(angle) * 50
                            )
                            
                            # Calculate connection activity
                            activity = node.network_activity
                            
                            # Pulse color based on activity
                            base_color = (0, 150, 255)
                            pulse = math.sin(time.time() * 10) * 0.5 + 0.5
                            color = tuple(int(c * (1 + activity * pulse)) for c in base_color)
                            color = (min(255, color[0])/255.0, min(255, color[1])/255.0, min(255, color[2])/255.0, node.alpha/255.0)
                            
                            thickness = 1 + int(activity * 3)
                            self.draw_line(screen_pos, end_pos, color, thickness)
                            
                            if self.show_stats:
                                conn_text = f"{conn.raddr.ip}:{conn.raddr.port}"
                                text = self.small_font.render(conn_text, True, tuple(int(c*255) for c in color[:3]))
                                self.text_surface.blit(text, (end_pos[0] + 5, end_pos[1] - 5))

        # Draw process connections (parent-child relationships)
        for node in self.nodes.values():
            if node.pid in visible_processes and node.parent_pid in visible_processes:
                parent = self.nodes[node.parent_pid]
                alpha = min(node.alpha, parent.alpha)
                color = (*node.color[:3], alpha)
                start_pos = self.world_to_screen(node.pos)
                end_pos = self.world_to_screen(parent.pos)
                self.draw_line(start_pos, end_pos, color, 1)

        # Draw nodes and collect text boxes
        new_processes_count = 0
        terminating_processes_count = 0
        text_boxes = []
        
        for node in self.nodes.values():
            if node.is_new:
                new_processes_count += 1
            if node.is_terminating:
                terminating_processes_count += 1
            
            # Only draw visible processes
            if node.pid in visible_processes:
                screen_pos = self.world_to_screen(node.pos)
                
                # Draw highlight for new processes
                if node.is_new:
                    glow_radius = node.radius + 5 + math.sin(time.time() * 5) * 2
                    glow_color = (255, 255, 100, node.alpha)
                    self.draw_circle(screen_pos[0], screen_pos[1], glow_radius, glow_color)

                # Scale node radius based on CPU and memory usage
                if self.show_stats and (node.cpu_percent > 0 or node.memory_percent > 0):
                    usage_scale = max(node.cpu_percent, node.memory_percent) / 100.0
                    scaled_radius = node.radius * (1 + usage_scale)
                else:
                    scaled_radius = node.radius

                # Draw node
                color = (*node.color[:3], node.alpha)
                self.draw_circle(screen_pos[0], screen_pos[1], scaled_radius, color)
                
                # Collect text box information
                if node.alpha > 128:
                    # Calculate text position
                    text_pos = (screen_pos[0] + node.text_offset[0] * self.zoom, 
                              screen_pos[1] + node.text_offset[1] * self.zoom)
                    
                    # Get text dimensions
                    name_text = f"{node.name} (PID: {node.pid})"
                    name_width, name_height = self.font.size(name_text)
                    
                    if self.show_stats:
                        stats_text = f"CPU: {node.cpu_percent:.1f}% MEM: {node.memory_percent:.1f}%"
                        stats_width = self.small_font.size(stats_text)[0]
                        width = max(name_width, stats_width)
                        height = name_height + self.small_font.get_height()
                    else:
                        width = name_width
                        height = name_height
                    
                    text_boxes.append({
                        'node': node,
                        'pos': text_pos,
                        'width': width,
                        'height': height,
                        'screen_pos': screen_pos
                    })

        # Adjust text positions to prevent overlap
        for i, box1 in enumerate(text_boxes):
            for box2 in text_boxes[i+1:]:
                # Check for overlap
                if (box1['pos'][0] < box2['pos'][0] + box2['width'] and
                    box1['pos'][0] + box1['width'] > box2['pos'][0] and
                    box1['pos'][1] < box2['pos'][1] + box2['height'] and
                    box1['pos'][1] + box1['height'] > box2['pos'][1]):
                    
                    # Adjust text offsets to prevent overlap
                    node1, node2 = box1['node'], box2['node']
                    center1 = box1['screen_pos']
                    center2 = box2['screen_pos']
                    
                    # Place text on opposite sides of nodes
                    angle = math.atan2(center2[1] - center1[1], center2[0] - center1[0])
                    
                    # Node 1 text placement
                    node1.text_offset = [
                        -math.cos(angle) * 50,
                        -math.sin(angle) * 50
                    ]
                    
                    # Node 2 text placement
                    node2.text_offset = [
                        math.cos(angle) * 50,
                        math.sin(angle) * 50
                    ]

        # Draw the text with adjusted positions
        for box in text_boxes:
            node = box['node']
            screen_pos = box['screen_pos']
            text_pos = (screen_pos[0] + node.text_offset[0] * self.zoom,
                       screen_pos[1] + node.text_offset[1] * self.zoom)
            
            text_color = (255, 255, 100) if node.is_new else (200, 200, 200)
            name_text = self.font.render(f"{node.name} (PID: {node.pid})", True, text_color)
            self.text_surface.blit(name_text, text_pos)
            
            if self.show_stats:
                stats_text = f"CPU: {node.cpu_percent:.1f}% MEM: {node.memory_percent:.1f}%"
                stats_surface = self.small_font.render(stats_text, True, (150, 150, 150))
                self.text_surface.blit(stats_surface, (text_pos[0], text_pos[1] + self.font.get_height()))

        # Draw stats
        stats_text = f"New: {new_processes_count} | Terminating: {terminating_processes_count} | Zoom: {self.zoom:.1f}x"
        stats_surface = self.font.render(stats_text, True, (200, 200, 200))
        self.text_surface.blit(stats_surface, (10, 10))

        # Draw controls help
        help_text = "Controls: Mouse Wheel = Zoom | Left Click + Drag = Pan | N = Toggle Network | S = Toggle Stats | ESC = Exit"
        help_surface = self.font.render(help_text, True, (150, 150, 150))
        self.text_surface.blit(help_surface, (10, self.height - 30))

        pygame.display.flip()

    def update_processes(self):
        current_time = time.time()
        
        try:
            # Always check for new/terminated processes
            if current_time - self.last_process_check >= self.process_check_interval:
                self.last_process_check = current_time
                try:
                    current_pids = set(p.pid for p in psutil.process_iter(['pid']))
                except Exception as e:
                    logger.error(f"Error getting process list: {str(e)}")
                    return
                
                # Quick check for new and terminated processes
                for pid in current_pids:
                    if pid not in self.nodes:
                        try:
                            proc = psutil.Process(pid)
                            with proc.oneshot():  # More efficient process info gathering
                                name = proc.name()
                                pos = self.get_spawn_position()
                                self.nodes[pid] = ProcessNode(pid, name, pos, current_time, is_initial=self.initial_startup)
                                self.total_processes_monitored += 1
                                if not self.initial_startup:
                                    logger.info(f"New process detected - PID: {pid}, Name: {name}")
                                    self.start_sound.play()
                        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as e:
                            logger.debug(f"Could not create process node for PID {pid}: {str(e)}")
                            continue

                # Handle terminated processes
                for pid in list(self.nodes.keys()):
                    if pid not in current_pids:
                        node = self.nodes[pid]
                        if node.alpha == 255:
                            logger.info(f"Process terminated - PID: {pid}, Name: {node.name}")
                            self.end_sound.play()
                            node.is_terminating = True
                            node.color = (1.0, 0.0, 0.0)
                        node.alpha = max(0, node.alpha - 10)
                        if node.alpha <= 0:
                            del self.nodes[pid]

            # Update a batch of processes every frame
            if self.nodes:
                pids = list(self.nodes.keys())
                start_idx = self.process_update_index
                end_idx = min(start_idx + self.processes_per_update, len(pids))
                
                for pid in pids[start_idx:end_idx]:
                    if pid not in self.nodes:  # Check if node still exists
                        continue
                        
                    node = self.nodes[pid]
                    try:
                        proc = psutil.Process(pid)
                        with proc.oneshot():  # Get all process info in one shot
                            node.cpu_percent = proc.cpu_percent()
                            node.memory_percent = proc.memory_percent()
                            node.network_connections = proc.connections()
                            
                            try:
                                new_io = proc.io_counters()
                                if node.net_io_counters:
                                    bytes_sent_delta = new_io.write_bytes - node.last_bytes_sent
                                    bytes_recv_delta = new_io.read_bytes - node.last_bytes_recv
                                    activity = (bytes_sent_delta + bytes_recv_delta) / (1024 * 1024)
                                    node.network_activity = min(1.0, activity)
                                node.net_io_counters = new_io
                                node.last_bytes_sent = new_io.write_bytes
                                node.last_bytes_recv = new_io.read_bytes
                            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                                node.network_activity *= 0.5
                    except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as e:
                        logger.debug(f"Could not update process info for PID {pid}: {str(e)}")
                        node.network_activity *= 0.5

                    node.time_alive += self.process_check_interval
                    if node.time_alive >= self.new_process_highlight_duration:
                        node.is_new = False

                # Update index for next frame
                self.process_update_index = end_idx if end_idx < len(pids) else 0

            self.initial_startup = False
            
        except Exception as e:
            logger.error(f"Error in update_processes: {str(e)}")

    def run(self):
        logger.info("Starting process visualization")
        try:
            # Do initial process population quickly
            current_pids = set(p.pid for p in psutil.process_iter())
            for pid in current_pids:
                try:
                    proc = psutil.Process(pid)
                    name = proc.name()
                    pos = self.get_spawn_position()
                    self.nodes[pid] = ProcessNode(pid, name, pos, time.time(), is_initial=True)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            self.last_frame_time = time.time()
            while self.running:
                try:
                    self.handle_input()
                    self.update_processes()
                    self.draw()
                    self.clock.tick(60)  # Cap at 60 FPS
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error during visualization: {str(e)}", exc_info=True)
        finally:
            logger.info(f"Shutting down - Total processes monitored: {self.total_processes_monitored}")
            pygame.quit()

if __name__ == "__main__":
    try:
        logger.info("Starting Procify application")
        visualizer = ProcessVisualizer()
        visualizer.run()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True) 