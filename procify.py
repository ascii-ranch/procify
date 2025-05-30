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
import threading

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
        self.target_pos = list(pos)
        self.velocity = [0.0, 0.0]
        self.creation_time = creation_time
        self.alpha = 255
        self.radius = 5
        self.is_new = not is_initial
        self.time_alive = 0
        self.is_terminating = False
        self.text_fade_time = 5.0  # Text fades after 5 seconds
        self.new_process_fade_time = 10.0  # New process status fades after 10 seconds
        # Brighter colors for new processes
        if self.is_new:
            self.color = (
                random.randint(180, 255) / 255.0,
                random.randint(180, 255) / 255.0,
                random.randint(180, 255) / 255.0
            )
            self.target_color = (0.0, 0.8, 0.0)  # Target green color
        else:
            self.color = (
                random.randint(50, 150) / 255.0,
                random.randint(50, 150) / 255.0,
                random.randint(50, 150) / 255.0
            )
            self.target_color = self.color
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
            
            # Get display information
            pygame.display.init()
            displays = pygame.display.get_desktop_sizes()
            
            # Set initial window size and minimum dimensions
            self.width = 1200
            self.height = 800
            self.min_width = 400
            self.min_height = 300
            
            # Always use primary monitor
            self.monitor_index = 0
            os.environ['SDL_VIDEO_WINDOW_POS'] = "50,50"
            
            self.screen = pygame.display.set_mode(
                (self.width, self.height),
                pgl.OPENGL | pgl.DOUBLEBUF | pgl.RESIZABLE
            )
            
            pygame.display.set_caption("Procify - Process Visualization")
            
            # Initialize OpenGL once
            self.update_viewport()
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_LINE_SMOOTH)
            glEnable(GL_POINT_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
            
            # Animation and timing
            self.clock = pygame.time.Clock()
            self.last_frame_time = pygame.time.get_ticks()
            self.last_process_update = pygame.time.get_ticks()
            self.process_update_interval = 2000  # 2 seconds between process updates
            self.target_fps = 20  # Target 20 FPS for smooth visualization
            self.frame_time = 1.0 / self.target_fps
            self.physics_update_interval = 1.0 / 30.0  # 30 Hz physics updates
            self.fixed_time_step = 1.0 / 30.0  # Fixed physics timestep (30 Hz)
            
            # Initialize other attributes
            self.nodes = {}
            self.visible_processes = set()
            self.window_drag = False
            self.drag_offset = (0, 0)
            self.titlebar_height = 30
            self.min_edge_length = 150
            self.repulsion_strength = 500
            self.attraction_strength = 0.03
            self.max_edge_length = 300
            self.animation_speed = 0.1
            
            # Create text surface with double buffering
            self.text_surfaces = [
                pygame.Surface((self.width, self.height), pygame.SRCALPHA),
                pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            ]
            self.current_text_surface = 0
            self.font = pygame.font.SysFont('Consolas', 12)  # Even smaller main font
            self.small_font = pygame.font.SysFont('Consolas', 10)  # Smaller stats font
            self.tiny_font = pygame.font.SysFont('Consolas', 9)  # Tiny font for controls
            
            self.generate_sound_effects()
            
            self.running = True
            self.show_network = True
            self.show_stats = False  # Stats off by default
            self.show_bar_graph = False
            self.show_titles = True  # Titles on by default
            self.hide_network_processes = False
            self.initial_startup = True
            self.total_processes_monitored = 0
            
            self.offset_x = 0
            self.offset_y = 0
            self.target_offset_x = 0
            self.target_offset_y = 0
            self.zoom = 1.0
            self.dragging = False
            self.last_mouse_pos = None
            self.drag_velocity = [0, 0]
            self.drag_smoothing = 0.8
            self.physics_smoothing = 0.15
            self.show_parent_edges = True
            
            # IP frequency tracking
            self.ip_frequencies = {}
            self.ip_bar_width = 200  # Width of the bar graph area
            self.ip_bar_margin = 5  # Margin between bars
            self.max_ip_bars = 20  # Maximum number of IPs to show in the bar graph
            
            # Physics parameters tuning
            self.max_force = 2.0
            self.damping = 0.92
            self.edge_color = (0.4, 0.4, 0.8, 0.6)  # Color for parent edges
            
            # Process data buffers for async updates
            self.process_data_buffer = {}
            self.process_update_in_progress = False
            self.process_update_thread = None
            self.process_data_lock = threading.Lock()
            
            # Performance monitoring
            self.frame_times = [1.0 / self.target_fps]
            self.max_frame_times = 60
            self.avg_frame_time = 1.0 / self.target_fps
            self.min_frame_time = 1.0 / 1000.0
            
            # OpenGL display lists for static elements
            self.static_display_lists = {}
            self.init_display_lists()
            
        except Exception as e:
            logger.error(f"Failed to initialize ProcessVisualizer: {str(e)}")
            raise

    def init_display_lists(self):
        """Initialize OpenGL display lists for static elements"""
        # Create display list for circle
        self.static_display_lists['circle'] = glGenLists(1)
        glNewList(self.static_display_lists['circle'], GL_COMPILE)
        segments = 32
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(0, 0)
        for i in range(segments + 1):
            angle = i * (2.0 * math.pi / segments)
            glVertex2f(math.cos(angle), math.sin(angle))
        glEnd()
        glEndList()

    def update_process_data_async(self):
        """Asynchronous process data update"""
        while self.running:
            try:
                if time.time() - self.last_process_update >= 2.0:  # 2 second interval
                    current_pids = set(p.pid for p in psutil.process_iter(['pid']))
                    new_process_data = {}
                    
                    for pid in current_pids:
                        try:
                            proc = psutil.Process(pid)
                            with proc.oneshot():
                                new_process_data[pid] = {
                                    'name': proc.name(),
                                    'parent_pid': proc.ppid(),
                                    'cpu_percent': proc.cpu_percent(),
                                    'memory_percent': proc.memory_percent(),
                                    'network_connections': proc.connections(),
                                    'is_new': pid not in self.nodes
                                }
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    
                    with self.process_data_lock:
                        self.process_data_buffer = new_process_data
                    
                    self.last_process_update = time.time()
                
                time.sleep(0.1)  # Sleep to prevent high CPU usage
                
            except Exception as e:
                logger.error(f"Error in process update thread: {str(e)}")
                time.sleep(1)  # Sleep longer on error

    def apply_process_updates(self):
        """Apply buffered process updates"""
        with self.process_data_lock:
            if not self.process_data_buffer:
                return
                
            new_process_data = self.process_data_buffer
            self.process_data_buffer = {}
        
        current_pids = set(new_process_data.keys())
        
        # Update nodes from buffer
        for pid, data in new_process_data.items():
            if pid not in self.nodes:
                pos = self.get_spawn_position()
                self.nodes[pid] = ProcessNode(pid, data['name'], pos, time.time(), is_initial=False)
                self.total_processes_monitored += 1
                if not self.initial_startup:
                    self.start_sound.play()
            
            # Update existing node data
            node = self.nodes[pid]
            node.parent_pid = data['parent_pid']
            node.cpu_percent = data['cpu_percent']
            node.memory_percent = data['memory_percent']
            node.network_connections = data['network_connections']
        
        # Handle terminated processes
        for pid in list(self.nodes.keys()):
            if pid not in current_pids:
                node = self.nodes[pid]
                if node.alpha == 255:
                    self.end_sound.play()
                    node.is_terminating = True
                    node.color = (1.0, 0.0, 0.0)
                node.alpha = max(0, node.alpha - 10)
                if node.alpha <= 0:
                    del self.nodes[pid]
        
        self.initial_startup = False

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

    def world_to_screen(self, pos):
        """Convert world coordinates to screen coordinates"""
        x = (pos[0] + self.offset_x) * self.zoom + self.width/2
        y = (pos[1] + self.offset_y) * self.zoom + self.height/2
        logger.debug(f"World coords {pos} to screen coords ({x}, {y})")
        return (x, y)

    def get_spawn_position(self) -> Tuple[float, float]:
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(100, 300)
        x = math.cos(angle) * radius
        y = math.sin(angle) * radius
        return (x, y)

    def handle_input(self):
        current_time = time.time()
        self.delta_time = min(current_time - self.last_frame_time, 0.1)
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
                elif event.key == pygame.K_b:
                    self.show_bar_graph = not self.show_bar_graph
                elif event.key == pygame.K_p:  # Toggle parent edges
                    self.show_parent_edges = not self.show_parent_edges
                elif event.key == pygame.K_t:  # Toggle titles
                    self.show_titles = not self.show_titles
                elif event.key == pygame.K_h:  # Toggle hiding network processes
                    self.hide_network_processes = not self.hide_network_processes
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    if mouse_pos[1] < self.titlebar_height:
                        # Instead of trying to move the window, we'll just ignore titlebar clicks
                        # as window management is platform-dependent
                        pass
                    else:
                        self.dragging = True
                        self.last_mouse_pos = event.pos
                        # Reset velocities when starting drag
                        self.drag_velocity = [0, 0]
                elif event.button == 4:
                    self.zoom *= 1.1
                elif event.button == 5:
                    self.zoom /= 1.1
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.window_drag = False
                    self.dragging = False
                    # Minimal momentum on release
                    self.drag_velocity = [x * 0.1 for x in self.drag_velocity]
            elif event.type == pygame.MOUSEMOTION:
                if self.window_drag:
                    # Remove window dragging functionality as it's not consistently supported
                    pass
                elif self.dragging:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    # Direct position update with minimal smoothing
                    self.target_offset_x = self.offset_x + dx / self.zoom
                    self.target_offset_y = self.offset_y + dy / self.zoom
                    self.drag_velocity = [dx / self.zoom * 0.1, dy / self.zoom * 0.1]  # Reduced velocity impact
                    self.last_mouse_pos = event.pos
            elif event.type == pygame.VIDEORESIZE:
                # Enforce minimum window size
                self.width = max(event.w, self.min_width)
                self.height = max(event.h, self.min_height)
                try:
                    self.screen = pygame.display.set_mode((self.width, self.height), 
                                                        pgl.OPENGL | pgl.DOUBLEBUF | pgl.RESIZABLE)
                    self.text_surfaces = [
                        pygame.Surface((self.width, self.height), pygame.SRCALPHA),
                        pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                    ]
                    self.update_viewport()
                except Exception as e:
                    logger.error(f"Error during resize: {str(e)}")
                    # Revert to previous size if resize fails
                    self.width = event.w
                    self.height = event.h
                    self.screen = pygame.display.set_mode((self.width, self.height), 
                                                        pgl.OPENGL | pgl.DOUBLEBUF | pgl.RESIZABLE)
                    self.text_surfaces = [
                        pygame.Surface((self.width, self.height), pygame.SRCALPHA),
                        pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                    ]
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
        """Draw a line with proper OpenGL state management"""
        try:
            # Convert screen coordinates to GL coordinates
            x1, y1 = start_pos
            x2, y2 = end_pos
            
            # Log line drawing attempt
            logger.debug(f"Drawing line from ({x1}, {y1}) to ({x2}, {y2}) with width {width}")
            
            # Check if line coordinates are valid
            if any(math.isnan(x) or math.isinf(x) for x in [x1, y1, x2, y2]):
                logger.error("Invalid line coordinates (NaN or Inf detected)")
                return
                
            if abs(x1) > 1e6 or abs(y1) > 1e6 or abs(x2) > 1e6 or abs(y2) > 1e6:
                logger.error("Line coordinates too large, might cause rendering issues")
                return
            
            glPushAttrib(GL_CURRENT_BIT | GL_LINE_BIT)
            glLineWidth(float(width))
            glColor4f(color[0], color[1], color[2], color[3] if len(color) > 3 else 1.0)
            
            glBegin(GL_LINES)
            glVertex2f(float(x1), float(y1))
            glVertex2f(float(x2), float(y2))
            glEnd()
            
            glPopAttrib()
            
            # Log successful line drawing
            logger.debug("Line drawn successfully")
            
        except Exception as e:
            logger.error(f"Error drawing line: {str(e)}", exc_info=True)

    def draw_bar_graph(self):
        """Draw IP frequency bar graph"""
        try:
            # Update IP frequencies
            self.ip_frequencies.clear()
            for node in self.nodes.values():
                for conn in node.network_connections:
                    if conn.status == 'ESTABLISHED' and conn.raddr:
                        ip = conn.raddr.ip
                        self.ip_frequencies[ip] = self.ip_frequencies.get(ip, 0) + 1

            if not self.ip_frequencies:
                return

            # Sort IPs by frequency
            sorted_ips = sorted(self.ip_frequencies.items(), key=lambda x: x[1], reverse=True)[:self.max_ip_bars]

            # Calculate dimensions
            bar_height = 15
            total_height = (bar_height + self.ip_bar_margin) * len(sorted_ips)
            start_y = self.height - 50 - total_height
            max_frequency = max(freq for _, freq in sorted_ips)

            # Draw background
            glColor4f(0.0, 0.0, 0.0, 0.7)
            glBegin(GL_QUADS)
            glVertex2f(10, start_y - 10)
            glVertex2f(self.ip_bar_width + 150, start_y - 10)
            glVertex2f(self.ip_bar_width + 150, start_y + total_height + 10)
            glVertex2f(10, start_y + total_height + 10)
            glEnd()

            # Draw bars and labels
            for i, (ip, frequency) in enumerate(sorted_ips):
                y = start_y + i * (bar_height + self.ip_bar_margin)
                width = (frequency / max_frequency) * self.ip_bar_width

                # Draw bar
                glColor4f(0.2, 0.6, 1.0, 0.8)
                glBegin(GL_QUADS)
                glVertex2f(10, y)
                glVertex2f(10 + width, y)
                glVertex2f(10 + width, y + bar_height)
                glVertex2f(10, y + bar_height)
                glEnd()

                # Draw text
                text = f"{ip}: {frequency}"
                text_surface = self.small_font.render(text, True, (200, 200, 200))
                self.text_surfaces[self.current_text_surface].blit(text_surface, (15 + self.ip_bar_width, y))

        except Exception as e:
            logger.error(f"Error drawing bar graph: {str(e)}", exc_info=True)

    def update_camera(self):
        """Smoothly update camera position with reduced lag"""
        if not self.dragging:
            # Minimal momentum when not dragging
            self.drag_velocity = [v * 0.8 for v in self.drag_velocity]  # Faster velocity decay
            self.target_offset_x += self.drag_velocity[0]
            self.target_offset_y += self.drag_velocity[1]
        
        # Quick response to target position
        dx = self.target_offset_x - self.offset_x
        dy = self.target_offset_y - self.offset_y
        
        # Limit maximum position change per frame
        max_delta = 20.0 / self.zoom  # Adjusted for zoom level
        dx = max(min(dx, max_delta), -max_delta)
        dy = max(min(dy, max_delta), -max_delta)
        
        self.offset_x += dx * self.drag_smoothing
        self.offset_y += dy * self.drag_smoothing

    def apply_layout_forces(self):
        """Apply force-directed layout with improved physics"""
        if not self.nodes:
            return
            
        forces = {pid: [0, 0] for pid in self.nodes}
        
        # Define boundaries with some elasticity
        boundary_margin = 100
        min_x = -self.width/2/self.zoom + boundary_margin
        max_x = self.width/2/self.zoom - boundary_margin
        min_y = -self.height/2/self.zoom + boundary_margin
        max_y = self.height/2/self.zoom - boundary_margin
        
        dt = self.fixed_time_step / 1000.0  # Convert to seconds
        
        # Apply forces with improved physics
        visible_nodes = [node for pid, node in self.nodes.items() if pid in self.visible_processes]
        for i, node1 in enumerate(visible_nodes):
            for node2 in visible_nodes[i+1:]:
                dx = node1.pos[0] - node2.pos[0]
                dy = node1.pos[1] - node2.pos[1]
                distance = math.sqrt(dx*dx + dy*dy) + 0.1
                
                # Smoother force falloff
                force = self.repulsion_strength / (distance * distance + 1.0)
                force = min(force, self.max_force)  # Cap maximum force
                
                fx = force * dx / distance
                fy = force * dy / distance
                
                forces[node1.pid][0] += fx
                forces[node1.pid][1] += fy
                forces[node2.pid][0] -= fx
                forces[node2.pid][1] -= fy
        
        # Apply attraction with smoother spring forces
        for node in visible_nodes:
            if node.parent_pid in self.nodes:
                parent = self.nodes[node.parent_pid]
                dx = parent.pos[0] - node.pos[0]
                dy = parent.pos[1] - node.pos[1]
                distance = math.sqrt(dx*dx + dy*dy) + 0.1
                
                if distance > self.min_edge_length:
                    # Smoother spring force
                    force = self.attraction_strength * math.pow(distance - self.min_edge_length, 0.8)
                    force = min(force, self.max_force)
                    
                    fx = force * dx / distance
                    fy = force * dy / distance
                    
                    forces[node.pid][0] += fx
                    forces[node.pid][1] += fy
        
        # Apply forces with improved damping and smoothing
        for pid, node in self.nodes.items():
            if not node.is_new and pid in self.visible_processes:
                # Apply damping to current velocity
                node.velocity[0] *= self.damping
                node.velocity[1] *= self.damping
                
                # Add force to velocity
                force_x = forces[pid][0]
                force_y = forces[pid][1]
                
                # Smooth force application
                node.velocity[0] += force_x * self.physics_smoothing * dt
                node.velocity[1] += force_y * self.physics_smoothing * dt
                
                # Update position with velocity
                node.pos[0] += node.velocity[0] * dt
                node.pos[1] += node.velocity[1] * dt
                
                # Elastic boundary constraints
                if node.pos[0] < min_x:
                    node.velocity[0] += (min_x - node.pos[0]) * 0.1
                elif node.pos[0] > max_x:
                    node.velocity[0] += (max_x - node.pos[0]) * 0.1
                    
                if node.pos[1] < min_y:
                    node.velocity[1] += (min_y - node.pos[1]) * 0.1
                elif node.pos[1] > max_y:
                    node.velocity[1] += (max_y - node.pos[1]) * 0.1

    def draw(self):
        """Render the current frame"""
        try:
            # Apply layout forces before drawing
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Create new text surface
            self.text_surfaces[self.current_text_surface].fill((0, 0, 0, 0))
            
            # Set up OpenGL for 2D drawing
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, self.width, self.height, 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            # Enable blending for transparency
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDisable(GL_DEPTH_TEST)
            
            # Draw title bar background
            glColor4f(0.2, 0.2, 0.2, 1.0)
            glBegin(GL_QUADS)
            glVertex2f(0, 0)
            glVertex2f(self.width, 0)
            glVertex2f(self.width, self.titlebar_height)
            glVertex2f(0, self.titlebar_height)
            glEnd()
            
            # First, identify processes that should be visible
            visible_processes = set()
            for node in self.nodes.values():
                has_network = any(conn.status == 'ESTABLISHED' for conn in node.network_connections)
                if self.hide_network_processes and has_network:
                    continue
                if (node.is_new or 
                    node.is_terminating or
                    node.alpha < 255 or 
                    (has_network and not self.hide_network_processes)):
                    visible_processes.add(node.pid)
                    # Also add parent if it exists
                    if node.parent_pid in self.nodes:
                        visible_processes.add(node.parent_pid)
                        # Add all parents up the chain
                        current_pid = node.parent_pid
                        while current_pid in self.nodes and self.nodes[current_pid].parent_pid in self.nodes:
                            current_pid = self.nodes[current_pid].parent_pid
                            visible_processes.add(current_pid)

            # Draw parent-child edges first
            if self.show_parent_edges:
                for node in self.nodes.values():
                    if node.pid in visible_processes and node.parent_pid in self.nodes:
                        parent = self.nodes[node.parent_pid]
                        if parent.pid in visible_processes:
                            start_pos = self.world_to_screen(node.pos)
                            end_pos = self.world_to_screen(parent.pos)
                            edge_alpha = min(node.alpha, parent.alpha) / 255.0
                            edge_color = (*self.edge_color[:3], edge_alpha)
                            self.draw_line(start_pos, end_pos, edge_color, 1)

            # Draw network connections
            if self.show_network:
                for node in self.nodes.values():
                    if node.pid in visible_processes:
                        screen_pos = self.world_to_screen(node.pos)
                        for conn in node.network_connections:
                            if conn.status == 'ESTABLISHED' and conn.raddr:
                                key = f"{conn.raddr.ip}:{conn.raddr.port}"
                                if key not in node.connection_endpoints:
                                    angle = random.uniform(0, 2 * math.pi)
                                    node.connection_endpoints[key] = angle
                                
                                angle = node.connection_endpoints[key]
                                end_pos = (
                                    screen_pos[0] + math.cos(angle) * 50,
                                    screen_pos[1] + math.sin(angle) * 50
                                )
                                
                                activity = node.network_activity
                                base_color = (0, 150, 255)
                                pulse = math.sin(time.time() * 10) * 0.5 + 0.5
                                color = tuple(int(c * (1 + activity * pulse)) for c in base_color)
                                color = (min(255, color[0])/255.0, min(255, color[1])/255.0, min(255, color[2])/255.0, node.alpha/255.0)
                                
                                thickness = 1 + int(activity * 3)
                                self.draw_line(screen_pos, end_pos, color, thickness)

            # Draw nodes
            for node in self.nodes.values():
                if node.pid in visible_processes:
                    screen_pos = self.world_to_screen(node.pos)
                    
                    # Draw highlight for new processes
                    if node.is_new:
                        glow_radius = node.radius + 5 + math.sin(time.time() * 5) * 2
                        glow_color = (1.0, 1.0, 0.4, node.alpha / 255.0)
                        self.draw_circle(screen_pos[0], screen_pos[1], glow_radius, glow_color)

                    # Draw node
                    color = (*node.color[:3], node.alpha / 255.0)
                    self.draw_circle(screen_pos[0], screen_pos[1], node.radius, color)

            # Draw IP frequency bar graph if enabled
            if self.show_network and self.show_bar_graph:
                self.draw_bar_graph()

            # Draw all text after OpenGL rendering
            glDisable(GL_DEPTH_TEST)
            
            # Draw title and FPS
            title_text = self.font.render("Procify - Process Visualization", True, (200, 200, 200))
            self.text_surfaces[self.current_text_surface].blit(title_text, (10, 5))

            # Draw FPS right next to title
            if self.show_stats:
                try:
                    current_fps = min(1000, 1.0 / max(self.min_frame_time, self.avg_frame_time))
                    fps_text = f"| FPS: {current_fps:.1f} | Frame Time: {self.avg_frame_time*1000:.1f}ms"
                    fps_surface = self.font.render(fps_text, True, (150, 150, 150))
                    self.text_surfaces[self.current_text_surface].blit(fps_surface, (title_text.get_width() + 20, 5))
                except Exception as e:
                    logger.error(f"Error rendering performance stats: {str(e)}")

            # Draw controls help with tiny font
            help_text = "Controls: Mouse Wheel = Zoom | Left Click + Drag = Pan | N = Toggle Network | S = Toggle Stats | B = Toggle Bar Graph | P = Toggle Parent Edges | T = Toggle Titles | H = Hide Network Processes | ESC = Exit"
            help_surface = self.tiny_font.render(help_text, True, (150, 150, 150))
            self.text_surfaces[self.current_text_surface].blit(help_surface, (10, self.height - 20))

            # Draw process names and stats
            for node in self.nodes.values():
                if node.pid in visible_processes and node.alpha > 128:
                    screen_pos = self.world_to_screen(node.pos)
                    
                    # Update node timing and colors
                    current_time = time.time()
                    node.time_alive = current_time - node.creation_time
                    
                    # Handle new process transition
                    if node.is_new and node.time_alive > node.new_process_fade_time:
                        node.is_new = False
                        # Interpolate color towards target green
                        for i in range(3):
                            node.color = tuple(
                                c + (node.target_color[i] - c) * 0.1
                                for i, c in enumerate(node.color)
                            )
                    
                    try:
                        # Only render text if show_titles is enabled or the process is new/terminating
                        should_show_text = (self.show_titles or 
                                         (node.is_new and node.time_alive < node.text_fade_time) or
                                         (node.is_terminating and node.alpha > 128))
                        
                        if should_show_text:
                            # Render process name with PID
                            name_color = (255, 0, 0) if node.is_terminating else (
                                (255, 255, 100) if node.is_new else (200, 200, 200)
                            )
                            name_text = self.font.render(f"{node.name} (PID: {node.pid})", True, name_color)
                            
                            # Calculate text position (to the right of the node)
                            text_x = int(screen_pos[0] + node.radius + 10)
                            text_y = int(screen_pos[1] - name_text.get_height() // 2)  # Vertically centered
                            
                            # Calculate text alpha for fading
                            text_alpha = 255
                            if node.is_new and node.time_alive > node.text_fade_time:
                                text_alpha = max(0, 255 * (1 - (node.time_alive - node.text_fade_time)))
                            elif node.is_terminating:
                                text_alpha = node.alpha
                            
                            # Draw text with background
                            text_bg = pygame.Surface((name_text.get_width() + 10, name_text.get_height() + 4), pygame.SRCALPHA)
                            text_bg.fill((0, 0, 0, int(text_alpha * 0.5)))
                            self.text_surfaces[self.current_text_surface].blit(text_bg, (text_x - 5, text_y - 2))
                            
                            # Apply alpha to text surface
                            text_surface = name_text.copy()
                            text_surface.set_alpha(int(text_alpha))
                            self.text_surfaces[self.current_text_surface].blit(text_surface, (text_x, text_y))
                            
                            # Draw stats if enabled (below the name)
                            if self.show_stats:
                                stats_text = f"CPU: {node.cpu_percent:.1f}% MEM: {node.memory_percent:.1f}%"
                                stats_surface = self.small_font.render(stats_text, True, (150, 150, 150))
                                stats_surface.set_alpha(int(text_alpha))
                                self.text_surfaces[self.current_text_surface].blit(stats_surface, (text_x, text_y + name_text.get_height()))
                                    
                    except Exception as e:
                        logger.error(f"Error rendering process text: {str(e)}")

            # Render all text at once
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            text_data = pygame.image.tostring(self.text_surfaces[self.current_text_surface], 'RGBA', True)
            glWindowPos2d(0, 0)
            glDrawPixels(self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            
            glEnable(GL_DEPTH_TEST)
            
            pygame.display.flip()
            
        except Exception as e:
            logger.error(f"Error in draw method: {str(e)}", exc_info=True)

    def run(self):
        """Main loop with optimized update cycles"""
        logger.info("Starting process visualization")
        try:
            # Start process update thread
            self.process_update_thread = threading.Thread(target=self.update_process_data_async, daemon=True)
            self.process_update_thread.start()
            
            # Initial process population
            current_pids = set(p.pid for p in psutil.process_iter())
            for pid in current_pids:
                try:
                    proc = psutil.Process(pid)
                    name = proc.name()
                    pos = self.get_spawn_position()
                    self.nodes[pid] = ProcessNode(pid, name, pos, time.time(), is_initial=True)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            physics_accumulator = 0.0
            last_time = time.time()

            while self.running:
                current_time = time.time()
                frame_time = current_time - last_time
                last_time = current_time
                
                # Cap frame time to prevent spiral
                frame_time = min(frame_time, 0.25)
                physics_accumulator += frame_time
                
                # Handle input (needs to be responsive)
                self.handle_input()
                
                # Apply process updates from buffer
                self.apply_process_updates()
                
                # Fixed timestep physics updates
                while physics_accumulator >= self.physics_update_interval:
                    self.apply_layout_forces()
                    self.update_camera()
                    physics_accumulator -= self.physics_update_interval
                
                # Render frame
                self.draw()
                
                # Maintain target frame rate
                self.clock.tick(self.target_fps)
                
                # Update performance metrics
                frame_time = max(self.min_frame_time, time.time() - current_time)
                self.frame_times.append(frame_time)
                if len(self.frame_times) > self.max_frame_times:
                    self.frame_times.pop(0)
                self.avg_frame_time = sum(self.frame_times) / len(self.frame_times)

        except Exception as e:
            logger.error(f"Error during visualization: {str(e)}", exc_info=True)
        finally:
            self.running = False
            if self.process_update_thread:
                self.process_update_thread.join(timeout=1.0)
            pygame.quit()

if __name__ == "__main__":
    try:
        logger.info("Starting Procify application")
        visualizer = ProcessVisualizer()
        visualizer.run()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True) 