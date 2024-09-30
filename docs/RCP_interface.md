# Event Bus + Direct Access Shared Memory Interface

The interface between RCP internal and external system is based on the data sharing medium of direct access memory, and
event handling via an event bus.

# Channels

| Topic               | Event Types                    |
|---------------------|--------------------------------|
| rcp:external_status | `data`                         |
| rcp:flagging        | `ready`, `data`, `ack`, `done` |
| rcp:calibration     | `ready`, `data`, `ack`, `done` |

## To Subscribe to a topic

```json
{
  "event": "subscribe",
  "topic": "{topic}"
}
```

## To Unsubscribe to a topic

```json
{
  "event": "unsubscribe",
  "topic": "{topic}"
}
```

## To publish data to a topic

```json
{
  "event": "publish",
  "topic": "{topic}",
  "type": "{type}",
  "data": "{data}"
}
```

This causes a message of the following form to be distributed to all subscribers:

```json
{
  "topic": "{topic}",
  "event": "{type}",
  "data": "{data}"
}
```

# Server event types

These are events that come from the server.

## Subscribed

Sent when a client subscribes to a topic.

```json
{
  "topic": "{topic}",
  "event": "subscribed"
}
```

## Unsubscribed

Sent when a client unsubscribes from a topic.

```json
{
  "topic": "{topic}",
  "event": "unsubscribed"
}
```

## Shutdown

Sent when server is shutting down.

```json
{
  "event": "shutdown"
}
```

# Status channel event types

These are events that are sent on the status channel by the external system. The intended recipient is the RCP internal.

## Data

```json
{
  "topic": "rcp:external_status",
  "event": "data",
  "data": {
    "timestamp": "{timestamp}",
    "status": "{status}",
    "node_ip": "{node_ip}"
  }
}
```

| Field     | Description                             |
|-----------|-----------------------------------------|
| timestamp | The UNIX time when status was measured. |
| status    | The status of the external system.      |
| node_ip   | The IP address of the external worker.  |


# Flagging channel event types

These are events that are sent on the flagging channel by the external system. This mediate communication between
internal and external systems for the flagging task.

## Ready

Sent from the external system to the RCP internal when the flagging system is ready to receive flagging data. After
receiving this, the external system worker will remain wait until it receives the `data` event on its node IP.

```json
{
  "topic": "rcp:flagging",
  "event": "ready",
  "data": {
    "node_ip": "{node_ip}",
    "vis_shm_name": "{vis_shm_name}",
    "weights_shm_name": "{weights_shm_name}",
    "flags_shm_name": "{flags_shm_name}"
  }
}
```

| Field            | Description                                     |
|------------------|-------------------------------------------------|
| node_ip          | The IP address of the external client.          |
| vis_shm_name     | The name of the shared memory for visibilities. |
| weights_shm_name | The name of the shared memory for weights.      |
| flags_shm_name   | The name of the shared memory for flags.        |

## Data

Sent by RCP internal when it has placed data in the Shared Memory for the external system to process on the given Node
IP.

```json
{
  "topic": "rcp:flagging",
  "event": "data",
  "data": {
    "node_ip": "{node_ip}"
  }
}
```

| Field   | Description                                              |
|---------|----------------------------------------------------------|
| node_ip | The IP address where the shared memory has been updated. |

## Ack

Sent by the external system to the RCP internal when it has received the data from the shared memory, and is processing
it. If this is not received within a given time frame then the RCP should assume something went wrong. The status events
can be used to get the status of the node in question. After receiving this, the RCP internal should not send any more
data until it has received the `done` event.

```json
{
  "topic": "rcp:flagging",
  "event": "ack",
  "data": {
    "node_ip": "{node_ip}"
  }
}
```

| Field   | Description                                                      |
|---------|------------------------------------------------------------------|
| node_ip | The IP address of external system acknowledging receipt of data. |

## Done

Sent by the external system to the RCP internal when it has finished processing the data from the shared memory. After
receiving this, the RCP internal can read the results from the shared memory. The flags shared memory will have
been updated.

```json
{
  "topic": "rcp:flagging",
  "event": "done",
  "data": {
    "node_ip": "{node_ip}"
  }
}
```

# Calibration channel event types

These are events that are sent on the calibration channel by the external system. The intended recipient is the RCP
internal.

## Ready

Sent from the external system to the RCP internal when the calibration system is ready to receive calibration data.
After receiving this, the external system worker will remain wait until it receives the `data` event on its node IP.

```json
{
  "topic": "rcp:calibration",
  "event": "ready",
  "data": {
    "node_ip": "{node_ip}",
    "vis_shm_name": "{vis_shm_name}",
    "weights_shm_name": "{weights_shm_name}",
    "flags_shm_name": "{flags_shm_name}"
  }
}
```

| Field            | Description                                     |
|------------------|-------------------------------------------------|
| node_ip          | The IP address of the external client.          |
| vis_shm_name     | The name of the shared memory for visibilities. |
| weights_shm_name | The name of the shared memory for weights.      |
| flags_shm_name   | The name of the shared memory for flags.        |

## Data

Sent by RCP internal when it has placed data in the Shared Memory for the external system to process on the given Node
IP.

```json
{
  "topic": "rcp:calibration",
  "event": "data",
  "data": {
    "node_ip": "{node_ip}"
  }
}
```

| Field   | Description                                              |
|---------|----------------------------------------------------------|
| node_ip | The IP address where the shared memory has been updated. |

## Ack

Sent by the external system to the RCP internal when it has received the data from the shared memory, and is processing
it. If this is not received within a given time frame then the RCP should assume something went wrong. The status events
can be used to get the status of the node in question. After receiving this, the RCP internal should not send any more
data until it has received the `done` event.

```json
{
  "topic": "rcp:calibration",
  "event": "ack",
  "data": {
    "node_ip": "{node_ip}"
  }
}
```

| Field   | Description                                                      |
|---------|------------------------------------------------------------------|
| node_ip | The IP address of external system acknowledging receipt of data. |

## Done

Sent by the external system to the RCP internal when it has finished processing the data from the shared memory. After
receiving this, the RCP internal can read the results from the shared memory. The visibility shared memory will have
been updated.

```json
{
  "topic": "rcp:calibration",
  "event": "done",
  "data": {
    "node_ip": "{node_ip}"
  }
}
```



