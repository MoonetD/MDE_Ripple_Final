# Web-Ripple: Hydration Monitoring and Suggestion System

Web-Ripple is a comprehensive hydration monitoring system designed to track water consumption for elderly individuals through video analysis and provide actionable suggestions to caregivers. The system analyzes videos of water glasses, food intake, and facial hydration indicators to generate personalized hydration recommendations.

## Project Structure

```
Web-Ripple/
├── app.py                      # Main Flask application
├── analyze_video_sample.py     # Video frame analysis
├── analyze_consumption.py      # Water consumption analysis
├── analyze_suggestion.py       # Hydration suggestion generation
├── custom-key.txt              # OpenAI API key (not tracked in git)
├── static/
│   ├── assets/                 # Input video storage
│   │   ├── video_{id}_input_glass.mp4  # Glass/water input videos
│   │   ├── video_{id}_input_food.mp4   # Food input videos
│   │   └── video_{id}_input_face.mp4   # Face input videos
│   ├── video_resp/             # Analyzed video outputs
│   │   ├── video_{id}_water_glass.mp4  # Processed glass videos
│   │   ├── video_{id}_food.mp4         # Processed food videos
│   │   └── video_{id}_face.mp4         # Processed face videos
│   ├── response_data/          # User-specific data
│   │   ├── users.json          # User information
│   │   ├── suggestions.json    # Combined suggestions
│   │   └── {id}/               # User-specific folders
│   │       ├── analysis_data.json      # Raw analysis data
│   │       ├── consumption_data.json   # Detailed consumption data
│   │       ├── consumption.json        # Simplified consumption data
│   │       ├── suggestions.json        # User-specific suggestions
│   │       ├── water_levels.png        # Water level graph
│   │       ├── analyzed_frames/        # Processed video frames
│   │       └── camera_frames/          # Raw video frames
│   └── images/                 # UI images and graphs
├── templates/                  # HTML templates
│   ├── base.html               # Base template
│   ├── index.html              # Home/leaderboard page
│   └── user_details.html       # User dashboard
└── README.md                   # This documentation
```

## Input Video Requirements

The system expects input videos to be stored in specific locations with specific naming conventions:

1. **Glass/Water Videos**: `/static/assets/video_{id}_input_glass.mp4`
   - These videos should show a glass of water being consumed
   - The system tracks the water level changes to determine consumption

2. **Food Videos**: `/static/assets/video_{id}_input_food.mp4`
   - These videos should show food being consumed
   - The system analyzes food with water content

3. **Face Videos**: `/static/assets/video_{id}_input_face.mp4`
   - These videos should show the person's face
   - The system analyzes facial features for hydration indicators

Where `{id}` is the user's unique identifier (e.g., 1001, 1002).

## Analysis Pipeline

### 1. Video Analysis (`analyze_video_sample.py`)

This script processes the input videos frame by frame to extract key information:

- **Glass Analysis**: Detects glasses, tracks water levels, and identifies drinking events
- **Food Analysis**: Identifies food items and estimates water content
- **Face Analysis**: Analyzes facial features for signs of dehydration

The script:
1. Extracts frames from the input videos
2. Processes each frame to detect objects of interest
3. Saves processed frames to `static/response_data/{id}/analyzed_frames/`
4. Saves raw frames to `static/response_data/{id}/camera_frames/`
5. Generates processed videos in `static/video_resp/`
6. Creates `analysis_data.json` with detailed frame-by-frame analysis

### 2. Consumption Analysis (`analyze_consumption.py`)

This script analyzes the data from `analysis_data.json` to calculate water consumption:

```bash
python3 analyze_consumption.py static/response_data 1002 static/response_data/1002
```

The script:
1. Reads `static/response_data/{id}/analysis_data.json`
2. Uses GPT-4 to analyze the sequence of frames and detect drinking events
3. Calculates total water consumed (as a percentage of glass capacity)
4. Converts percentage to actual milliliters (assuming 200ml is 100%)
5. Generates a water level graph (`water_levels.png`)
6. Saves detailed consumption data to `consumption_data.json`
7. Saves simplified consumption data to `consumption.json` in the format:
   ```json
   {
     "water": 120  // ml of water consumed
   }
   ```

### 3. Suggestion Generation (`analyze_suggestion.py`)

This script generates personalized hydration suggestions based on consumption data:

```bash
python3 analyze_suggestion.py static/response_data/1002/consumption_info.json static/response_data/1002
```

The script:
1. Reads the consumption data from `consumption.json`
2. Uses OpenAI's GPT-4 to generate contextually relevant suggestions
3. Formats the suggestions with priority levels, icons, and action items
4. Saves user-specific suggestions to `static/response_data/{id}/suggestions.json`
5. Updates the main `static/response_data/suggestions.json` file with all users' suggestions

## Web Application (`app.py`)

The Flask application serves as the user interface and orchestrates the analysis pipeline:

### Routes

1. **Home/Leaderboard** (`/`):
   - Displays all users with their hydration statistics
   - Shows high-priority suggestions for caregivers
   - Allows sorting users by various metrics

2. **User Details** (`/user/<ripple_id>`):
   - Shows detailed hydration information for a specific user
   - Displays water sources chart, consumption metrics, and personalized suggestions
   - Provides links to view analyzed videos
   - Offers both caregiver and technical views

3. **Video Pages**:
   - `/food_video/<ripple_id>`: Food analysis video page
   - `/face_video/<ripple_id>`: Face analysis video page
   - `/glass_video/<ripple_id>`: Glass/water analysis video page

4. **Analysis Endpoints**:
   - `/analyze_video/<ripple_id>/<video_type>`: Triggers video analysis
   - `/analyze_consumption/<ripple_id>`: Triggers consumption analysis
   - `/generate_suggestions/<ripple_id>`: Triggers suggestion generation

### Key Functions

1. **Data Loading**:
   - `load_users_data()`: Loads user information from `users.json`
   - `load_suggestions()`: Loads suggestions from `suggestions.json`

2. **User Management**:
   - `create_user()`: Creates a new user with default values

3. **Consumption Data Handling**:
   - Reads user-specific consumption data from `consumption.json`
   - Uses default values (2% of highest value) when data is missing
   - Determines hydration status based on face analysis data

4. **Suggestion Handling**:
   - Reads user-specific suggestions from `suggestions.json`
   - Handles empty or invalid files gracefully

5. **Video Processing**:
   - Serves processed videos from `static/video_resp/`
   - Triggers analysis pipelines when requested

## Running the Application

1. **Prerequisites**:
   - Python 3.x
   - Flask
   - OpenAI API key (in `custom-key.txt`)
   - Required Python packages (matplotlib, OpenAI, etc.)

2. **Start the server**:
   ```bash
   python3 app.py
   ```

3. **Access the application**:
   - Open a web browser and navigate to `http://localhost:5001`

4. **Analyze videos for a user**:
   - Place input videos in the correct locations
   - Navigate to the user's page
   - Use the "Analyze" buttons to trigger the analysis pipeline

## Data Flow

1. Input videos are stored in `static/assets/`
2. Videos are analyzed by `analyze_video_sample.py`
3. Analysis data is stored in `static/response_data/{id}/analysis_data.json`
4. Consumption is calculated by `analyze_consumption.py`
5. Consumption data is stored in `static/response_data/{id}/consumption.json`
6. Suggestions are generated by `analyze_suggestion.py`
7. Suggestions are stored in `static/response_data/{id}/suggestions.json`
8. The web application displays all data in a user-friendly interface

## Security Considerations

- The OpenAI API key is stored in `custom-key.txt` and read at runtime
- User data is stored in JSON files and not encrypted
- No authentication is implemented in this version

## Future Enhancements

- Database integration for more robust data storage
- User authentication and role-based access control
- Mobile application for caregivers
- Real-time notifications for critical hydration issues
- Integration with smart water bottles and other IoT devices
