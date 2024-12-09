import json
import os

import ray

from dsa2000_fm.forward_models.streaming.single_kernel.process_actor import get_node_ip

# Directory to store Prometheus target files
TARGETS_DIR = "/etc/prometheus/dynamic_targets"  # Replace with actual path
PROMETHEUS_PORT = 8090


def get_ray_nodes():
    """Fetch the list of all Ray nodes and their IPs."""
    try:
        # Initialize a connection to the Ray head node
        ray.init(address="auto", ignore_reinit_error=True)

        # Fetch the list of all nodes
        node_info = ray.nodes()
        return [{"ip": node["NodeManagerAddress"], "name": node["NodeName"]} for node in node_info if node["Alive"]]
    except Exception as e:
        print(f"Error fetching Ray nodes: {e}")
        return []


def write_target_file(node_id):
    """Write or update JSON files for each Ray node."""
    os.makedirs(TARGETS_DIR, exist_ok=True)
    node_ip = get_node_ip(node_id)
    target_file = os.path.join(TARGETS_DIR, f"{node_ip}.json")

    if os.path.exists(target_file):
        return  # Skip if the file already exists

    # Define target JSON content
    target_content = [
        {
            "targets": [f"{node_ip}:{PROMETHEUS_PORT}"],
            "labels": {"job": "ray_workers", "node_name": node_id},
        }
    ]

    # Write the JSON file
    with open(target_file, "w") as f:
        json.dump(target_content, f, indent=2)
    print(f"Updated target file: {target_file}")


def main():
    print("Scraping Ray nodes and writing Prometheus target files...")
    nodes = get_ray_nodes()
    for node in nodes:
        node_id = node["NodeID"]
        write_target_file(node_id)


if __name__ == "__main__":
    main()
