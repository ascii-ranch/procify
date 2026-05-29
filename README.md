# Procify

A real-time process visualizer that maps your running OS processes as an interactive node graph.

![Procify screenshot](procify_v1.PNG)

Each node is a running process. Lines connect children to their parent processes. New processes glow yellow briefly when they appear, and nodes fade out as processes terminate.

## Features

- Live process tree — updates every 500ms
- Parent-child connections drawn between related processes
- New process highlighting with animated glow
- Pan and zoom to explore dense graphs
- Stats bar showing active process count and zoom level

## Requirements

- Python 3.12+
- Linux or WSL2 with a display (X11 / WSLg)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install psutil pygame numpy
```

## Run

```bash
python3 procify.py
```

## Controls

| Input | Action |
|---|---|
| Mouse wheel | Zoom in/out |
| Left click + drag | Pan |
| ESC | Exit |

```
      ___           ___           ___
     /\  \         /\  \         /\  \          ___         ___
    /::\  \       /::\  \       /::\  \        /\  \       /\  \
   /:/\:\  \     /:/\ \  \     /:/\:\  \       \:\  \      \:\  \
  /::\~\:\  \   _\:\~\ \  \   /:/  \:\  \      /::\__\     /::\__\
 /:/\:\ \:\__\ /\ \:\ \ \__\ /:/__/ \:\__\  __/:/\/__/  __/:/\/__/
 \/__\:\/:/  / \:\ \:\ \/__/ \:\  \  \/__/ /\/:/  /    /\/:/  /
      \::/  /   \:\ \:\__\    \:\  \       \::/__/     \::/__/
      /:/  /     \:\/:/  /     \:\  \       \:\__\      \:\__\
     /:/  /       \::/  /       \:\__\       \/__/       \/__/
     \/__/         \/__/         \/__/
      ___           ___           ___           ___           ___
     /\  \         /\  \         /\__\         /\  \         /\__\
    /::\  \       /::\  \       /::|  |       /::\  \       /:/  /
   /:/\:\  \     /:/\:\  \     /:|:|  |      /:/\:\  \     /:/__/
  /::\~\:\  \   /::\~\:\  \   /:/|:|  |__   /:/  \:\  \   /::\  \ ___
 /:/\:\ \:\__\ /:/\:\ \:\__\ /:/ |:| /\__\ /:/__/ \:\__\ /:/\:\  /\__\
 \/_|::\/:/  / \/__\:\/:/  / \/__|:|/:/  / \:\  \  \/__/ \/__\:\/:/  /
    |:|::/  /       \::/  /      |:/:/  /   \:\  \            \::/  /
    |:|\/__/        /:/  /       |::/  /     \:\  \           /:/  /
    |:|  |         /:/  /        /:/  /       \:\__\         /:/  /
     \|__|         \/__/         \/__/         \/__/         \/__/
```
