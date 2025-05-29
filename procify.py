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
            
            # Get display information
            pygame.display.init()
            displays = pygame.display.get_desktop_sizes()
            
            # Use second monitor if available
            if len(displays) > 1:
                self.monitor_index = 1
                monitor_x = displays[0][0]
                self.width = min(1200, displays[1][0] - 100)
                self.height = min(800, displays[1][1] - 100)
                os.environ['SDL_VIDEO_WINDOW_POS'] = f"{monitor_x + 50},{50}"
            else:
                self.monitor_index = 0
                self.width = min(1200, displays[0][0] - 100)
                self.height = min(800, displays[0][1] - 100)
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
            self.animation_accumulator = 0
            self.fixed_time_step = 16.67  # ~60 FPS for physics
            
            # Initialize other attributes
            self.nodes = {}
            self.visible_processes = set()
            self.window_drag = False
            self.drag_offset = (0, 0)
            self.titlebar_height = 30
            self.min_edge_length = 150
            self.repulsion_strength = 1000  # Reduced from 2000
            self.attraction_strength = 0.05  # Reduced from 0.1
            self.max_edge_length = 300
            self.animation_speed = 0.1  # Reduced from 0.2
            
            # Create text surface
            self.text_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
            
            self.generate_sound_effects()
            
            self.running = True
            self.show_network = True
            self.show_stats = True
            self.initial_startup = True
            self.total_processes_monitored = 0
            
            self.offset_x = 0
            self.offset_y = 0
            self.zoom = 1.0
            self.dragging = False
            self.last_mouse_pos = None
            
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

    def apply_layout_forces(self):
        """Apply force-directed layout with fixed timestep"""
        if not self.nodes:
            return
            
        forces = {pid: [0, 0] for pid in self.nodes}
        
        # Define boundaries
        boundary_margin = 100
        min_x = -self.width/2/self.zoom + boundary_margin
        max_x = self.width/2/self.zoom - boundary_margin
        min_y = -self.height/2/self.zoom + boundary_margin
        max_y = self.height/2/self.zoom - boundary_margin
        
        # Apply forces with fixed timestep
        dt = self.fixed_time_step / 1000.0  # Convert to seconds
        
        # Apply repulsion between visible nodes only
        visible_nodes = [node for pid, node in self.nodes.items() if pid in self.visible_processes]
        for i, node1 in enumerate(visible_nodes):
            for node2 in visible_nodes[i+1:]:
                dx = node1.pos[0] - node2.pos[0]
                dy = node1.pos[1] - node2.pos[1]
                distance = math.sqrt(dx*dx + dy*dy) + 0.1
                
                force = self.repulsion_strength / (distance * distance)
                fx = force * dx / distance
                fy = force * dy / distance
                
                forces[node1.pid][0] += fx
                forces[node1.pid][1] += fy
                forces[node2.pid][0] -= fx
                forces[node2.pid][1] -= fy
        
        # Apply attraction for connected nodes
        for node in visible_nodes:
            if node.parent_pid in self.nodes:
                parent = self.nodes[node.parent_pid]
                dx = parent.pos[0] - node.pos[0]
                dy = parent.pos[1] - node.pos[1]
                distance = math.sqrt(dx*dx + dy*dy) + 0.1
                
                if distance > self.min_edge_length:
                    force = self.attraction_strength * (distance - self.min_edge_length)
                    forces[node.pid][0] += force * dx / distance
                    forces[node.pid][1] += force * dy / distance
        
        # Apply forces with damping
        damping = 0.8
        for pid, node in self.nodes.items():
            if not node.is_new and pid in self.visible_processes:
                force_x = forces[pid][0] * damping
                force_y = forces[pid][1] * damping
                
                # Limit maximum force
                force_magnitude = math.sqrt(force_x*force_x + force_y*force_y)
                if force_magnitude > 5:
                    force_x *= 5/force_magnitude
                    force_y *= 5/force_magnitude
                
                # Update position with fixed timestep
                node.pos[0] += force_x * self.animation_speed * dt
                node.pos[1] += force_y * self.animation_speed * dt
                
                # Apply boundary constraints
                node.pos[0] = max(min_x, min(max_x, node.pos[0]))
                node.pos[1] = max(min_y, min(max_y, node.pos[1]))

    def update_processes(self):
        """Update process data at fixed intervals"""
        current_time = pygame.time.get_ticks()
        
        if current_time - self.last_process_update >= self.process_update_interval:
            self.last_process_update = current_time
            try:
                current_pids = set(p.pid for p in psutil.process_iter(['pid']))
                
                # Handle new processes
                for pid in current_pids:
                    if pid not in self.nodes:
                        try:
                            proc = psutil.Process(pid)
                            with proc.oneshot():
                                name = proc.name()
                                pos = self.get_spawn_position()
                                self.nodes[pid] = ProcessNode(pid, name, pos, current_time, is_initial=self.initial_startup)
                                self.total_processes_monitored += 1
                                if not self.initial_startup:
                                    logger.info(f"New process detected - PID: {pid}, Name: {name}")
                                    self.start_sound.play()
                        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
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

                # Update process info
                for pid, node in self.nodes.items():
                    try:
                        proc = psutil.Process(pid)
                        with proc.oneshot():
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
                            except:
                                node.network_activity *= 0.5
                    except:
                        node.network_activity *= 0.5
                        continue

                    node.time_alive += self.process_update_interval / 1000.0
                    if node.time_alive >= 5.0:  # 5 seconds highlight duration
                        node.is_new = False
                        
            except Exception as e:
                logger.error(f"Error updating processes: {str(e)}")
            
            self.initial_startup = False

    def draw(self):
        try:
            # Apply layout forces before drawing
            self.apply_layout_forces()
            
            # Clear both OpenGL and text surface
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Create a new surface each frame
            self.text_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            self.text_surface.fill((0, 0, 0, 0))
            
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
                if (node.is_new or 
                    node.is_terminating or
                    node.alpha < 255 or 
                    any(conn.status == 'ESTABLISHED' for conn in node.network_connections)):
                    visible_processes.add(node.pid)
                    # Also add parent if it exists
                    if node.parent_pid in self.nodes:
                        visible_processes.add(node.parent_pid)
                        # Add all parents up the chain
                        current_pid = node.parent_pid
                        while current_pid in self.nodes and self.nodes[current_pid].parent_pid in self.nodes:
                            current_pid = self.nodes[current_pid].parent_pid
                            visible_processes.add(current_pid)

            # Draw process connections (parent-child relationships)
            logger.debug("Drawing parent-child relationships")
            for node in self.nodes.values():
                if node.pid in visible_processes and node.parent_pid in self.nodes:
                    parent = self.nodes[node.parent_pid]
                    start_pos = self.world_to_screen(node.pos)
                    end_pos = self.world_to_screen(parent.pos)
                    alpha = min(node.alpha, parent.alpha) / 255.0
                    color = (*parent.color[:3], alpha)
                    self.draw_line(start_pos, end_pos, color, 2)
                    logger.debug(f"Drew edge from PID {node.pid} to parent PID {node.parent_pid}")

            # Draw network connections
            if self.show_network:
                logger.debug("Drawing network connections")
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
                                logger.debug(f"Drew network connection for PID {node.pid} to {key}")

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

            # Now draw all text after OpenGL rendering
            glDisable(GL_DEPTH_TEST)
            
            # Draw title
            title_text = self.font.render("Procify - Process Visualization (Drag to move)", True, (200, 200, 200))
            self.text_surface.blit(title_text, (10, 5))
            
            # Draw process names and stats
            for node in self.nodes.values():
                if node.pid in visible_processes and node.alpha > 128:
                    screen_pos = self.world_to_screen(node.pos)
                    try:
                        # Render process name with PID
                        name_color = (255, 255, 100) if node.is_new else (200, 200, 200)
                        name_text = self.font.render(f"{node.name} (PID: {node.pid})", True, name_color)
                        
                        # Calculate text position (above the node)
                        text_x = int(screen_pos[0] - name_text.get_width() // 2)
                        text_y = int(screen_pos[1] - node.radius - 20)
                        
                        # Draw text with background
                        text_bg = pygame.Surface((name_text.get_width() + 10, 25), pygame.SRCALPHA)
                        text_bg.fill((0, 0, 0, 128))
                        self.text_surface.blit(text_bg, (text_x - 5, text_y - 5))
                        self.text_surface.blit(name_text, (text_x, text_y))
                        
                        # Draw stats if enabled
                        if self.show_stats:
                            stats_text = f"CPU: {node.cpu_percent:.1f}% MEM: {node.memory_percent:.1f}%"
                            stats_surface = self.small_font.render(stats_text, True, (150, 150, 150))
                            stats_x = int(screen_pos[0] - stats_surface.get_width() // 2)
                            self.text_surface.blit(stats_surface, (stats_x, text_y + 20))
                            
                        # Draw connection info
                        if self.show_network:
                            for conn in node.network_connections:
                                if conn.status == 'ESTABLISHED' and conn.raddr:
                                    key = f"{conn.raddr.ip}:{conn.raddr.port}"
                                    angle = node.connection_endpoints[key]
                                    end_pos = (
                                        screen_pos[0] + math.cos(angle) * 50,
                                        screen_pos[1] + math.sin(angle) * 50
                                    )
                                    conn_text = f"{conn.raddr.ip}:{conn.raddr.port}"
                                    text = self.small_font.render(conn_text, True, (100, 200, 255))
                                    self.text_surface.blit(text, (end_pos[0] + 5, end_pos[1] - 5))
                                    
                    except Exception as e:
                        logger.error(f"Error rendering process text: {str(e)}")

            # Draw stats
            stats_text = f"New: {len(self.nodes) - len(self.nodes) for node in self.nodes.values() if not node.is_new} | Terminating: {len(self.nodes) for node in self.nodes.values() if node.is_terminating} | Zoom: {self.zoom:.1f}x"
            stats_surface = self.font.render(stats_text, True, (200, 200, 200))
            self.text_surface.blit(stats_surface, (10, 10))

            # Draw controls help
            help_text = "Controls: Mouse Wheel = Zoom | Left Click + Drag = Pan | N = Toggle Network | S = Toggle Stats | ESC = Exit"
            help_surface = self.font.render(help_text, True, (150, 150, 150))
            self.text_surface.blit(help_surface, (10, self.height - 30))

            # Render all text at once
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            text_data = pygame.image.tostring(self.text_surface, 'RGBA', True)
            glWindowPos2d(0, 0)  # Use glWindowPos2d instead of glRasterPos2d
            glDrawPixels(self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            
            glEnable(GL_DEPTH_TEST)
            
            pygame.display.flip()
            
        except Exception as e:
            logger.error(f"Error in draw method: {str(e)}", exc_info=True)

    def run(self):
        """Main loop with fixed timestep for physics"""
        logger.info("Starting process visualization")
        try:
            # Initial process population
            current_pids = set(p.pid for p in psutil.process_iter())
            for pid in current_pids:
                try:
                    proc = psutil.Process(pid)
                    name = proc.name()
                    pos = self.get_spawn_position()
                    self.nodes[pid] = ProcessNode(pid, name, pos, pygame.time.get_ticks(), is_initial=True)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            while self.running:
                frame_time = pygame.time.get_ticks()
                dt = frame_time - self.last_frame_time
                self.last_frame_time = frame_time
                
                # Handle input
                self.handle_input()
                
                # Update processes at fixed interval
                self.update_processes()
                
                # Accumulate time for physics updates
                self.animation_accumulator += dt
                
                # Update physics with fixed timestep
                while self.animation_accumulator >= self.fixed_time_step:
                    self.apply_layout_forces()
                    self.animation_accumulator -= self.fixed_time_step
                
                # Render
                self.draw()
                
                # Cap framerate
                self.clock.tick(60)

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