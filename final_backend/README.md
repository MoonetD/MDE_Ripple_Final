# Video Recording API

A FastAPI-based API for starting and processing video recordings.

## Overview

This API provides two main endpoints:
- `/video/start_record`: Initiates video recording
- `/video/process_video`: Stops recording and processes the video

## Features

- Simple REST API for video recording control
- FastAPI with automatic OpenAPI documentation
- CORS support for cross-origin requests
- Health check endpoint

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository or download the source code

2. Navigate to the project directory:
   ```bash
   cd final_backend
   ```

3. Create a virtual environment (optional but recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the API

Start the API server with:

```bash
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Root Endpoint

- **URL**: `/`
- **Method**: `GET`
- **Description**: Returns a welcome message and available endpoints
- **Response Example**:
  ```json
  {
    "message": "Welcome to the Video Recording API",
    "docs": "/docs",
    "endpoints": {
      "start_record": "/video/start_record",
      "process_video": "/video/process_video"
    }
  }
  ```

### Health Check

- **URL**: `/health`
- **Method**: `GET`
- **Description**: Returns the health status of the API
- **Response Example**:
  ```json
  {
    "status": "healthy"
  }
  ```

### Start Recording

- **URL**: `/video/start_record`
- **Method**: `POST`
- **Description**: Starts video recording
- **Response Example**:
  ```json
  {
    "success": true
  }
  ```

### Process Video

- **URL**: `/video/process_video`
- **Method**: `POST`
- **Description**: Stops recording and processes the video (includes a 2-second pause)
- **Response Example**:
  ```json
  {
    "success": true
  }
  ```

## Usage Examples

### Using curl

Start recording:
```bash
curl -X POST http://localhost:8000/video/start_record
```

Process video:
```bash
curl -X POST http://localhost:8000/video/process_video
```

### Using Python requests

```python
import requests

# Start recording
response = requests.post("http://localhost:8000/video/start_record")
print(response.json())  # {"success": true}

# Process video
response = requests.post("http://localhost:8000/video/process_video")
print(response.json())  # {"success": true}
```

## Notes

- This API is a simplified example and does not include actual video recording functionality.
- In a real-world application, you would need to implement the actual video recording and processing logic.
- The API includes a 2-second pause in the process_video endpoint to simulate processing time.

## License

This project is open source and available under the [MIT License](LICENSE).