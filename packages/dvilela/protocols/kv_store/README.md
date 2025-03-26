# Key-Value Storage Protocol

## Description

This is a protocol for key-value storage.

## Specification

```yaml
---
name: kv_store
author: dvilela
version: 0.1.0
description: A protocol for simple key-value storage.
license: Apache-2.0
aea_version: '>=1.0.0, <2.0.0'
protocol_specification_id: dvilela/kv_store:0.1.0
speech_acts:
  read_request:
    keys: pt:list[pt:str]
  read_response:
    data: pt:dict[pt:str, pt:str]
  create_or_update_request:
    data: pt:dict[pt:str, pt:str]
  success:
    message: pt:str
  error:
    message: pt:str
...
---
initiation: [read_request, create_or_update_request]
reply:
  read_request: [read_response, error]
  read_response: []
  create_or_update_request: [success, error]
  success: []
  error: []
termination: [read_response, success, error]
roles: {skill, connection}
end_states: [successful]
keep_terminal_state_dialogues: false
```

## Links

