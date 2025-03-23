#!/usr/bin/env python
"""Scrape Ray nodes and write Prometheus target files."""

import json
import logging
import os

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from dsa2000_common.common.ray_utils import get_head_node_id, get_node_ip

from dsa2000_common.common.logging import dsa_logger as logger

# Directory to store Prometheus target files
TARGETS_DIR = "/etc/prometheus/dynamic_targets"  # Replace with actual path
PROMETHEUS_PORT = 8090


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
    logger.info(f"Updated target file: {target_file}")


def scrape_task():
    logger.info("Scraping Ray nodes and writing Prometheus target files...")
    nodes = ray.nodes()
    for node in nodes:
        node_id = node["NodeID"]
        write_target_file(node_id)


def main():
    ray.init(address="auto")
    placement_node_id = get_head_node_id()
    task_options = {
        "scheduling_strategy": NodeAffinitySchedulingStrategy(placement_node_id, soft=False)}
    ray.get(ray.remote(scrape_task).options(**task_options).remote())


if __name__ == "__main__":
    main()
