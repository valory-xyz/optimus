# GenericMirrorDB Connection

The `GenericMirrorDB` connection is a flexible, generic Python-based REST API client designed to interact seamlessly with backend REST APIs. It leverages asynchronous programming to ensure high efficiency, robustness, and scalability, particularly suited for applications requiring continuous communication with REST services.

## Overview

This module provides generic methods (`create_function`, `read_function`, `update_function`, `delete_function`) that abstract the typical CRUD operations (Create, Read, Update, Delete) performed via REST APIs. It simplifies managing HTTP requests by encapsulating complex logic such as connection handling, retries with exponential backoff, error handling, and API endpoint validation.

## Why GenericMirrorDB?

GenericMirrorDB was created to:

- **Reduce redundancy:** Provides common REST operations in a generic way, reducing repetitive code.
- **Improve maintainability:** Centralized error handling and retries make the codebase easier to maintain.
- **Flexible and extendable:** Easy to configure and extend, supporting custom endpoints and configurations.
- **Efficient handling:** Asynchronous requests and exponential backoff ensure resilience and high availability.

## Key Components

### Core CRUD Operations
- `create_function`: Executes POST requests to create resources.
- `read_function`: Executes GET requests to retrieve resources.
- `update_function`: Executes PUT requests to modify resources.
- `delete_function`: Executes DELETE requests to remove resources.

### Configuration and Customization
- `update_api_key`: Update API credentials dynamically.
- `update_config`: Update multiple configuration parameters simultaneously.
- `get_config`: Retrieve the current configuration state.

### Endpoint Management
- `register_endpoint`: Dynamically add custom endpoints to the validated set.
- Endpoint validation ensures security by verifying endpoints against predefined patterns.

### Retry Logic
Implemented via `retry_with_exponential_backoff` decorator:
- Automatically retries API calls upon encountering rate limits or transient errors.
- Delay between retries increases exponentially, reducing load on backend APIs.

## Usage Example

Here's how you can use GenericMirrorDB:

```python
# Assume an instantiated connection object
await connection.update_api_key("your_api_key")
await connection.register_endpoint("api/resource")

# Reading a resource
response = await connection.read_function(
    method_name="fetch_resource",
    endpoint="api/resource/resource_id"
)
print(response)
```

## Workflow

1. **Initialization**: The connection is established asynchronously.
2. **Validation**: Endpoint requests are validated against registered patterns.
3. **Request Handling**: Generic CRUD methods execute HTTP requests.
4. **Retry on Failure**: On rate limit errors, requests retry automatically.
5. **Response Management**: Successful responses or structured errors are returned.

## Error Handling

Errors encountered during HTTP requests or invalid requests are returned with clear, structured messages, enabling easier debugging and issue resolution.

## Dependencies

- Python 3.7+
- `aiohttp` for asynchronous HTTP operations
- `certifi` for secure SSL connections

## License

This project is licensed under the Apache License, Version 2.0.

