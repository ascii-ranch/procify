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
        self.velocity = [0, 0]
        self.creation_time = creation_time
        self.alpha = 255
        self.radius = 5
        self.is_new = not is_initial  # Don't mark initial processes as new
        self.time_alive = 0  # Track how long the process has been visible
        self.color = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
        self.parent_pid = None
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.network_connections = []
        self.last_update = time.time()
        self.net_io_counters = None
        self.last_bytes_sent = 0
        self.last_bytes_recv = 0
        self.network_activity = 0.0  # Scale from 0 to 1 indicating recent activity
        
        try:
            proc = psutil.Process(pid)
            self.parent_pid = proc.ppid()
            self.cpu_percent = proc.cpu_percent()
            self.memory_percent = proc.memory_percent()
            self.network_connections = proc.connections()
            try:
                self.net_io_counters = proc.io_counters()
                self.last_bytes_sent = self.net_io_counters.write_bytes
                self.last_bytes_recv = self.net_io_counters.read_bytes
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            logger.info(f"{'Initial' if is_initial else 'New'} process node created - PID: {pid}, Name: {name}, Parent PID: {self.parent_pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Could not get process info for {pid} ({name}): {str(e)}")
            
    def update_stats(self):
        """Update process statistics"""
        try:
            proc = psutil.Process(self.pid)
            current_time = time.time()
            if current_time - self.last_update >= 1.0:  # Update every second
                self.cpu_percent = proc.cpu_percent()
                self.memory_percent = proc.memory_percent()
                self.network_connections = proc.connections()
                
                # Update network activity
                try:
                    new_io = proc.io_counters()
                    if self.net_io_counters:
                        bytes_sent_delta = new_io.write_bytes - self.last_bytes_sent
                        bytes_recv_delta = new_io.read_bytes - self.last_bytes_recv
                        # Calculate activity level (0-1) based on bytes transferred
                        activity = (bytes_sent_delta + bytes_recv_delta) / (1024 * 1024)  # MB/s
                        self.network_activity = min(1.0, activity)
                    self.net_io_counters = new_io
                    self.last_bytes_sent = new_io.write_bytes
                    self.last_bytes_recv = new_io.read_bytes
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self.network_activity *= 0.5  # Decay activity if we can't measure it
                
                self.last_update = current_time
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

class ProcessVisualizer:
    def __init__(self):
        logger.info("Initializing ProcessVisualizer")
        pygame.init()
        pygame.mixer.init(44100, -16, 2, 1024)
        
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Procify - Process Visualization")
        
        # Generate sound effects
        self.generate_sound_effects()
        
        self.clock = pygame.time.Clock()
        self.nodes: Dict[int, ProcessNode] = {}
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.last_process_check = time.time()
        self.process_check_interval = 0.2  # Increased refresh rate from 0.5 to 0.2 seconds
        self.running = True
        self.total_processes_monitored = 0
        
        # Camera/view controls
        self.offset_x = 0
        self.offset_y = 0
        self.zoom = 1.0
        self.dragging = False
        self.last_mouse_pos = None
        self.new_process_highlight_duration = 5.0  # Seconds to highlight new processes
        
        # Display options
        self.show_network = True
        self.show_stats = True
        
        # Track initial startup
        self.initial_startup = True
        
        logger.info(f"ProcessVisualizer initialized with window size: {self.width}x{self.height}")

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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                logger.info("Received quit signal")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    logger.info("Received escape key - shutting down")
                elif event.key == pygame.K_n:  # Toggle network connections
                    self.show_network = not self.show_network
                elif event.key == pygame.K_s:  # Toggle stats
                    self.show_stats = not self.show_stats
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.dragging = True
                    self.last_mouse_pos = event.pos
                elif event.button == 4:  # Mouse wheel up
                    self.zoom *= 1.1
                elif event.button == 5:  # Mouse wheel down
                    self.zoom /= 1.1
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    self.dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    self.offset_x += dx / self.zoom
                    self.offset_y += dy / self.zoom
                    self.last_mouse_pos = event.pos

    def update_processes(self):
        current_time = time.time()
        if current_time - self.last_process_check < self.process_check_interval:
            return

        self.last_process_check = current_time
        current_pids = set()

        # Update time_alive and stats for existing processes
        for node in self.nodes.values():
            node.time_alive += self.process_check_interval
            if node.time_alive >= self.new_process_highlight_duration:
                node.is_new = False
            node.update_stats()

        # Check for new processes
        for proc in psutil.process_iter(['pid', 'name', 'create_time']):
            try:
                pid = proc.info['pid']
                current_pids.add(pid)
                
                if pid not in self.nodes:
                    name = proc.info['name']
                    pos = self.get_spawn_position()
                    # Mark as initial process if we're just starting up
                    self.nodes[pid] = ProcessNode(pid, name, pos, current_time, is_initial=self.initial_startup)
                    self.total_processes_monitored += 1
                    # Only play sound for new processes after initial startup
                    if not self.initial_startup:
                        logger.info(f"New process detected - PID: {pid}, Name: {name}")
                        self.start_sound.play()

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.debug(f"Access error while monitoring process: {str(e)}")
                continue

        # Clear initial startup flag after first process check
        self.initial_startup = False

    def update_node_positions(self):
        for node in self.nodes.values():
            # Only update positions for new or fading nodes
            if node.is_new or node.alpha < 255:
                dx = node.target_pos[0] - node.pos[0]
                dy = node.target_pos[1] - node.pos[1]
                
                node.velocity[0] = dx * 0.1
                node.velocity[1] = dy * 0.1
                
                node.pos[0] += node.velocity[0]
                node.pos[1] += node.velocity[1]

    def draw(self):
        self.screen.fill((0, 0, 0))

        # First, identify processes that should be visible
        visible_processes = set()
        for node in self.nodes.values():
            if (node.is_new or 
                node.alpha < 255 or 
                any(conn.status == 'ESTABLISHED' for conn in node.network_connections)):
                visible_processes.add(node.pid)
                # Also add parent if it exists
                if node.parent_pid in self.nodes:
                    visible_processes.add(node.parent_pid)

        # Draw network connections if enabled
        if self.show_network:
            # First pass: collect all connection endpoints
            connection_map = {}
            for node in self.nodes.values():
                for conn in node.network_connections:
                    if conn.status == 'ESTABLISHED':
                        if conn.laddr and conn.raddr:
                            key = (conn.laddr.port, conn.raddr.port)
                            connection_map[key] = (node, conn)

            # Second pass: draw connections between processes
            for (lport, rport), (node, conn) in connection_map.items():
                # Try to find the other end of the connection
                for other_node in self.nodes.values():
                    for other_conn in other_node.network_connections:
                        if (other_conn.status == 'ESTABLISHED' and 
                            other_conn.laddr and other_conn.raddr and
                            other_conn.laddr.port == rport and
                            other_conn.raddr.port == lport):
                            # Found a matching connection
                            start_pos = self.world_to_screen(node.pos)
                            end_pos = self.world_to_screen(other_node.pos)
                            
                            # Calculate connection activity (max of both ends)
                            activity = max(node.network_activity, other_node.network_activity)
                            
                            # Pulse color based on activity
                            base_color = (0, 150, 255)
                            pulse = math.sin(time.time() * 10) * 0.5 + 0.5  # Oscillate between 0 and 1
                            color = tuple(int(c * (1 + activity * pulse)) for c in base_color)
                            color = (min(255, color[0]), min(255, color[1]), min(255, color[2]), node.alpha)
                            
                            # Draw connection line with varying thickness based on activity
                            thickness = 1 + int(activity * 3)
                            pygame.draw.line(self.screen, color, start_pos, end_pos, thickness)
                            
                            # Draw connection info if stats are enabled
                            if self.show_stats and activity > 0:
                                mid_x = (start_pos[0] + end_pos[0]) / 2
                                mid_y = (start_pos[1] + end_pos[1]) / 2
                                activity_text = f"{activity * 100:.1f}%"
                                text = self.small_font.render(activity_text, True, color)
                                self.screen.blit(text, (mid_x, mid_y))

            # Draw unmatched connections as short lines
            for node in self.nodes.values():
                unmatched_conns = [c for c in node.network_connections 
                                 if c.status == 'ESTABLISHED' and 
                                 not any(n != node and any(oc.status == 'ESTABLISHED' and 
                                                         oc.laddr and c.raddr and
                                                         oc.laddr.port == c.raddr.port 
                                                         for oc in n.network_connections)
                                       for n in self.nodes.values())]
                
                if unmatched_conns:
                    start_pos = self.world_to_screen(node.pos)
                    for conn in unmatched_conns:
                        if conn.raddr:
                            # Draw a short line for external connections
                            angle = random.uniform(0, 2 * math.pi)
                            end_pos = (start_pos[0] + math.cos(angle) * 50,
                                     start_pos[1] + math.sin(angle) * 50)
                            
                            # Pulse color based on activity
                            activity = node.network_activity
                            base_color = (0, 150, 255)
                            pulse = math.sin(time.time() * 10) * 0.5 + 0.5
                            color = tuple(int(c * (1 + activity * pulse)) for c in base_color)
                            color = (min(255, color[0]), min(255, color[1]), min(255, color[2]), node.alpha)
                            
                            thickness = 1 + int(activity * 3)
                            pygame.draw.line(self.screen, color, start_pos, end_pos, thickness)
                            
                            if self.show_stats:
                                conn_text = f"{conn.raddr.ip}:{conn.raddr.port}"
                                text = self.small_font.render(conn_text, True, color)
                                self.screen.blit(text, (end_pos[0] + 5, end_pos[1] - 5))

        # Draw process connections (parent-child relationships)
        for node in self.nodes.values():
            if node.pid in visible_processes and node.parent_pid in visible_processes:
                parent = self.nodes[node.parent_pid]
                alpha = min(node.alpha, parent.alpha)
                color = (*node.color[:3], alpha)
                start_pos = self.world_to_screen(node.pos)
                end_pos = self.world_to_screen(parent.pos)
                pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

        # Draw nodes
        new_processes_count = 0
        terminating_processes_count = 0
        
        for node in self.nodes.values():
            if node.is_new:
                new_processes_count += 1
            if node.alpha < 255:
                terminating_processes_count += 1
            
            # Only draw visible processes
            if node.pid in visible_processes:
                screen_pos = self.world_to_screen(node.pos)
                
                # Draw highlight for new processes
                if node.is_new:
                    glow_radius = node.radius + 5 + math.sin(time.time() * 5) * 2
                    glow_color = (255, 255, 100, node.alpha)
                    pygame.draw.circle(self.screen, glow_color,
                                     (int(screen_pos[0]), int(screen_pos[1])),
                                     int(glow_radius * self.zoom))

                # Scale node radius based on CPU and memory usage
                if self.show_stats and (node.cpu_percent > 0 or node.memory_percent > 0):
                    usage_scale = max(node.cpu_percent, node.memory_percent) / 100.0
                    scaled_radius = node.radius * (1 + usage_scale)
                else:
                    scaled_radius = node.radius

                # Draw node
                color = (*node.color[:3], node.alpha)
                pygame.draw.circle(self.screen, color,
                                 (int(screen_pos[0]), int(screen_pos[1])),
                                 int(scaled_radius * self.zoom))
                
                # Draw process info
                if node.alpha > 128:
                    text_color = (255, 255, 100) if node.is_new else (200, 200, 200)
                    name_text = self.font.render(f"{node.name} (PID: {node.pid})", True, text_color)
                    self.screen.blit(name_text, (screen_pos[0] + 10 * self.zoom, screen_pos[1] - 10 * self.zoom))
                    
                    if self.show_stats:
                        stats_text = f"CPU: {node.cpu_percent:.1f}% MEM: {node.memory_percent:.1f}%"
                        stats_surface = self.small_font.render(stats_text, True, (150, 150, 150))
                        self.screen.blit(stats_surface, (screen_pos[0] + 10 * self.zoom, screen_pos[1] + 5 * self.zoom))

        # Draw stats
        stats_text = f"New: {new_processes_count} | Terminating: {terminating_processes_count} | Zoom: {self.zoom:.1f}x"
        stats_surface = self.font.render(stats_text, True, (200, 200, 200))
        self.screen.blit(stats_surface, (10, 10))

        # Draw controls help
        help_text = "Controls: Mouse Wheel = Zoom | Left Click + Drag = Pan | N = Toggle Network | S = Toggle Stats | ESC = Exit"
        help_surface = self.font.render(help_text, True, (150, 150, 150))
        self.screen.blit(help_surface, (10, self.height - 30))

        pygame.display.flip()

    def run(self):
        logger.info("Starting process visualization")
        try:
            while self.running:
                self.handle_input()
                self.update_processes()
                self.update_node_positions()
                self.draw()
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