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

# Set up logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

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
    def __init__(self, pid: int, name: str, pos: Tuple[float, float], creation_time: float):
        self.pid = pid
        self.name = name
        self.pos = list(pos)
        self.target_pos = list(pos)
        self.velocity = [0, 0]
        self.creation_time = creation_time
        self.alpha = 255
        self.radius = 5
        self.is_new = True  # Flag for new processes
        self.time_alive = 0  # Track how long the process has been visible
        self.color = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
        self.parent_pid = None
        try:
            self.parent_pid = psutil.Process(pid).ppid()
            logger.info(f"New process node created - PID: {pid}, Name: {name}, Parent PID: {self.parent_pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Could not get parent PID for process {pid} ({name}): {str(e)}")

class ProcessVisualizer:
    def __init__(self):
        logger.info("Initializing ProcessVisualizer")
        pygame.init()
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Procify - Process Visualization")
        
        self.clock = pygame.time.Clock()
        self.nodes: Dict[int, ProcessNode] = {}
        self.font = pygame.font.Font(None, 24)
        self.last_process_check = time.time()
        self.process_check_interval = 0.5
        self.running = True
        self.total_processes_monitored = 0
        
        # Camera/view controls
        self.offset_x = 0
        self.offset_y = 0
        self.zoom = 1.0
        self.dragging = False
        self.last_mouse_pos = None
        self.new_process_highlight_duration = 5.0  # Seconds to highlight new processes
        
        logger.info(f"ProcessVisualizer initialized with window size: {self.width}x{self.height}")

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

        # Update time_alive for existing processes
        for node in self.nodes.values():
            node.time_alive += self.process_check_interval
            if node.time_alive >= self.new_process_highlight_duration:
                node.is_new = False

        # Check for new processes
        for proc in psutil.process_iter(['pid', 'name', 'create_time']):
            try:
                pid = proc.info['pid']
                current_pids.add(pid)
                
                if pid not in self.nodes:
                    name = proc.info['name']
                    pos = self.get_spawn_position()
                    self.nodes[pid] = ProcessNode(pid, name, pos, current_time)
                    self.total_processes_monitored += 1
                    logger.info(f"New process detected - PID: {pid}, Name: {name}")

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.debug(f"Access error while monitoring process: {str(e)}")
                continue

        # Remove terminated processes
        for pid in list(self.nodes.keys()):
            if pid not in current_pids:
                node = self.nodes[pid]
                if node.alpha == 255:
                    logger.info(f"Process terminated - PID: {pid}, Name: {node.name}")
                node.alpha = max(0, node.alpha - 10)
                if node.alpha <= 0:
                    del self.nodes[pid]

    def update_node_positions(self):
        for node in self.nodes.values():
            dx = node.target_pos[0] - node.pos[0]
            dy = node.target_pos[1] - node.pos[1]
            
            node.velocity[0] = dx * 0.1
            node.velocity[1] = dy * 0.1
            
            node.pos[0] += node.velocity[0]
            node.pos[1] += node.velocity[1]

    def draw(self):
        self.screen.fill((0, 0, 0))

        # Draw connections first
        for node in self.nodes.values():
            if node.parent_pid in self.nodes:
                parent = self.nodes[node.parent_pid]
                alpha = min(node.alpha, parent.alpha)
                color = (*node.color[:3], alpha)
                start_pos = self.world_to_screen(node.pos)
                end_pos = self.world_to_screen(parent.pos)
                pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

        # Draw nodes
        new_processes_count = 0
        for node in self.nodes.values():
            if node.is_new:
                new_processes_count += 1
            
            screen_pos = self.world_to_screen(node.pos)
            
            # Draw highlight for new processes
            if node.is_new:
                glow_radius = node.radius + 5 + math.sin(time.time() * 5) * 2
                glow_color = (255, 255, 100, node.alpha)
                pygame.draw.circle(self.screen, glow_color,
                                 (int(screen_pos[0]), int(screen_pos[1])),
                                 int(glow_radius * self.zoom))

            # Draw node
            color = (*node.color[:3], node.alpha)
            pygame.draw.circle(self.screen, color,
                             (int(screen_pos[0]), int(screen_pos[1])),
                             int(node.radius * self.zoom))
            
            # Draw process name
            if node.alpha > 128:
                text_color = (255, 255, 100) if node.is_new else (200, 200, 200)
                text = self.font.render(f"{node.name[:15]}", True, text_color)
                text_pos = (screen_pos[0] + 10 * self.zoom, screen_pos[1] - 10 * self.zoom)
                self.screen.blit(text, text_pos)

        # Draw stats
        stats_text = f"New Processes: {new_processes_count} | Total Active: {len(self.nodes)} | Zoom: {self.zoom:.1f}x"
        stats_surface = self.font.render(stats_text, True, (200, 200, 200))
        self.screen.blit(stats_surface, (10, 10))

        # Draw controls help
        help_text = "Controls: Mouse Wheel = Zoom | Left Click + Drag = Pan | ESC = Exit"
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