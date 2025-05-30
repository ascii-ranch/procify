from pyvis.network import Network
import random
import json

# Create a new network with physics enabled
net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")

# Configure physics for smooth interactions
net.force_atlas_2based()
physics = {
    "enabled": True,
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {
        "gravitationalConstant": -50,
        "centralGravity": 0.01,
        "springLength": 100,
        "springConstant": 0.08,
        "damping": 0.4,
        "avoidOverlap": 1
    },
    "minVelocity": 0.75,
    "maxVelocity": 50,
}

# Create the options dictionary and convert it to a proper JSON string
options = {
    "physics": physics,
    "interaction": {
        "dragNodes": True,
        "dragView": True,
        "hideEdgesOnDrag": False,
        "hideNodesOnDrag": False,
        "hover": True,
        "navigationButtons": True,
        "selectable": True,
        "zoomView": True
    }
}
net.set_options(json.dumps(options))

# Generate some dummy data
num_nodes = 20
node_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Add nodes with random positions and colors
for i in range(num_nodes):
    net.add_node(
        i,
        label=f"Node {i}",
        color=random.choice(node_colors),
        size=25,
        title=f"Node {i} details"  # Tooltip on hover
    )

# Add some random edges between nodes
for i in range(num_nodes):
    # Connect each node to 2-4 random other nodes
    num_connections = random.randint(2, 4)
    for _ in range(num_connections):
        target = random.randint(0, num_nodes-1)
        if target != i:  # Avoid self-loops
            net.add_edge(i, target, color="#666666", width=2)

# Generate the HTML file
net.show("network.html", notebook=False)

print("Network visualization has been generated! Open 'network.html' in your browser to view it.") 