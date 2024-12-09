Example of dynamic targets for Prometheus. Put into a json file.

```json
[
  {
    "targets": ["worker1:8090", "worker2:8090", "worker3:8090"],
    "labels": {
      "job": "ray_workers"
    }
  }
]
```