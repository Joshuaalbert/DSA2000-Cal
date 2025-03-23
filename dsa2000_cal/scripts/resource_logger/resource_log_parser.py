import re
from collections import defaultdict
from datetime import datetime

import pylab as plt

# Define the regex pattern to match the log line.
# Expected log format:
# 2025-03-19 15:47:00,123  node_name [RESOURCE_TYPE]: value_with_units
pattern = re.compile(
    r'(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?)\s{2}'
    r'(?P<node>\S+)\s+\[(?P<resource_type>.+?)\]:\s+(?P<value>.+)$'
)


def extract_float(value_str):
    """
    Extract a float from a string that contains units.
    Returns None if the value is "N/A" or if no numeric value is found.
    """
    if value_str.strip().upper() == "N/A":
        return None
    # Search for a number (integer or float) in the string.
    match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", value_str)
    if match:
        return float(match.group())
    return None


def parse_log_line(line):
    """
    Parse a single log line and return a tuple (datetime, node, resource_type, value)
    where value is a float (or None if not available).
    """
    match = pattern.match(line)
    if match:
        dt_str = match.group("datetime")
        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S,%f")
        except ValueError:
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        node = match.group("node")
        resource_type = match.group("resource_type")
        value_str = match.group("value")
        numeric_value = extract_float(value_str)
        return dt, node, resource_type, numeric_value
    return None


def parse_log_file(filepath):
    """
    Parse the entire log file and return a list of tuples:
    (datetime, node, resource_type, value)
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            parsed = parse_log_line(line.strip())
            if parsed:
                records.append(parsed)
    return records


def plot_resource_usage(records):
    # Group records by resource type
    records_by_type = defaultdict(list)
    for dt, node, resource_type, value in records:
        records_by_type[(node, resource_type)].append((dt, value))

    # Plot each resource type
    fig, axs = plt.subplots(len(records_by_type), 1, figsize=(10, 5 * len(records_by_type)), sharex=True)
    nodes = sorted(set(node for node, _ in records_by_type))
    resources = sorted(set(resource_type for _, resource_type in records_by_type))
    node_colours = {node: plt.cm.get_cmap("tab10")(i) for i, node in enumerate(nodes)}
    resource_axes = {res: i for i, res in enumerate(resources)}
    for i, ((node, resource_type), items) in enumerate(records_by_type.items()):
        colour = node_colours[node]
        datetimes, values = zip(*items)
        ax = axs[resource_axes[resource_type]]
        ax.plot(datetimes, values, label=node, color=colour, marker='o', linestyle='-')
        ax.set_ylabel(resource_type)
        ax.set_title(resource_type)
        ax.legend()
        ax.grid(True)
    axs[-1].set_xlabel("Time")
    fig.savefig('resource_usage.png')
    plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Parse resource log file')
    parser.add_argument('--logfile', type=str, default='resource_usage.log', help='Path to the log file')
    args = parser.parse_args()

    records = parse_log_file(args.logfile)
    for record in records:
        print(record)
    plot_resource_usage(records)


if __name__ == '__main__':
    main()
