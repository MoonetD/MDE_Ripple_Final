import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from openai import OpenAI
import sys

# Initialize OpenAI client
with open('custom-key.txt', 'r') as f:
    api_key = f.read().strip()
client = OpenAI(api_key=api_key)

def parse_timestamp(timestamp_str):
    return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")

def analyze_sequence_with_gpt(frames):
    # Convert frames to a chronological sequence for analysis
    sequence = []
    sorted_frames = sorted(frames.items(), key=lambda x: x[0])
    
    for frame_id, frame_data in sorted_frames:
        if frame_data.get('has_glass', False):
            sequence.append({
                'timestamp': frame_data['timestamp'],
                'water_percentage': frame_data.get('water_percentage', 0),
                'glass_visible': frame_data.get('glass_visible', True),
                'glass_characteristics': frame_data.get('glass_characteristics', ''),
                'appears_different': frame_data.get('appears_different', False)
            })
    
    # Create prompt for GPT-4
    prompt = (
        "Analyze this sequence of video frames to identify drinking events. Return ONLY a JSON array of events, with no additional text or explanation.\n\n"
        "IMPORTANT ANALYSIS RULES:\n"
        "1. When water level increases slightly, it's likely measurement stabilization, NOT a refill\n"
        "2. When glass disappears and reappears with lower level, that's a strong drinking indicator\n"
        "3. Focus on net decreases in water level\n"
        "4. Small fluctuations (±5%) should be ignored as measurement noise\n"
        "5. Only count significant drops that are part of the same drinking sequence\n\n"
        "A drinking event is detected when:\n"
        "1. Vision-based indicators:\n"
        "   - Glass tilts >20° with hand gripping\n"
        "   - Glass disappears (moves up) and returns with lower level\n"
        "   - Water surface shows clear disturbance with hand contact\n\n"
        "2. Water level indicators:\n"
        "   - Net decrease >5% in water level\n"
        "   - Same glass is being tracked\n"
        "   - Changes occur within expected drinking timeframe\n\n"
        "For each REAL drinking event, include in JSON:\n"
        "- timestamp: when significant drop started\n"
        "- water_level_before: stable level before drop\n"
        "- water_level_after: stable level after drop\n"
        "- amount_consumed: net decrease (ignore temporary increases)\n"
        "- glass_characteristics: glass description\n"
        "- confidence: 0-100 based on indicators present\n"
        "- detection_method: which criteria were met\n"
        "- change_location: where change occurred\n\n"
        "Example response format:\n"
        "[{\n"
        "  \"timestamp\": \"20250501_012345_678901\",\n"
        "  \"water_level_before\": 80,\n"
        "  \"water_level_after\": 60,\n"
        "  \"amount_consumed\": 20,\n"
        "  \"glass_characteristics\": \"Clear cylindrical glass\",\n"
        "  \"confidence\": 90,\n"
        "  \"detection_method\": \"Water level drop + glass disappearance\",\n"
        "  \"change_location\": \"moved out of frame\"\n"
        "}]\n\n"
        "Frame sequence:\n"
        f"{json.dumps(sequence, indent=2)}"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0,
            max_tokens=1000
        )
        
        # Parse the response to extract drinking events
        analysis = response.choices[0].message.content
        print("\nGPT Analysis:\n", analysis)
        
        # Extract drinking events from the analysis
        drinking_events = []
        try:
            # Look for JSON array in the response
            if '[' in analysis and ']' in analysis:
                start = analysis.find('[')
                end = analysis.rfind(']') + 1
                events_json = json.loads(analysis[start:end])
                if isinstance(events_json, list):
                    # Filter out events with small water level changes
                    drinking_events = []
                    for event in events_json:
                        if event.get('amount_consumed', 0) > 5:  # Only include significant drops
                            # If water level temporarily increased, focus on the net decrease
                            if event.get('water_level_after', 0) > event.get('water_level_before', 0):
                                continue
                            drinking_events.append(event)
                elif isinstance(events_json, dict):
                    drinking_events = [events_json]
        except json.JSONDecodeError:
            print("Could not parse GPT response as JSON")
            drinking_events = []
        
        return drinking_events
        
    except Exception as e:
        print(f"Error analyzing sequence with GPT: {e}")
        return []

def analyze_water_consumption(dir, id, save_dir):
    # Load the analysis data
    with open(f"{dir}/{id}/analysis_data.json", 'r') as f:
        data = json.load(f)
    
    print("\nAnalyzing frames...")
    # Extract frames with water level measurements
    water_levels = []
    prev_percentage = None
    for timestamp, frame_data in sorted(data['frames'].items()):
        if frame_data.get('has_glass', False):
            current_percentage = frame_data.get('water_percentage', 0)
            if prev_percentage is not None:
                if abs(current_percentage - prev_percentage) > 5:
                    print(f"Water level change: {prev_percentage}% -> {current_percentage}%")
            
            water_levels.append({
                'timestamp': timestamp,
                'water_percentage': current_percentage,
                'glass_characteristics': frame_data.get('glass_characteristics'),
                'appears_different': frame_data.get('appears_different', False),
                'glass_visible': frame_data.get('glass_visible', True)
            })
            prev_percentage = current_percentage
        else:
            print(f"No glass visible at {timestamp}")
    
    # Sort by timestamp
    water_levels.sort(key=lambda x: x['timestamp'])
    
    # Use GPT to analyze the sequence and detect drinking events
    print("\nAnalyzing sequence with GPT...")
    drinking_events = analyze_sequence_with_gpt(data['frames'])
    
    # Calculate total water consumed
    total_water_consumed = sum(event.get('amount_consumed', 0) for event in drinking_events)
    
    print("\nWater Consumption Analysis:")
    print("-" * 26)
    print(f"Total water consumed across all glasses: {total_water_consumed}% of glass capacity")
    print(f"Number of drinking events detected: {len(drinking_events)}")
    
    print("\nPer-glass consumption:")
    glass_consumption = {}
    for event in drinking_events:
        glass_type = event.get('glass_characteristics', 'Unknown glass')
        if glass_type not in glass_consumption:
            glass_consumption[glass_type] = 0
        glass_consumption[glass_type] += event.get('amount_consumed', 0)
    
    for glass_type, amount in glass_consumption.items():
        print(f"  {glass_type}: {amount}% consumed")
    
    print("\nDetailed drinking events:")
    for event in drinking_events:
        print(f"  At {event['timestamp']}:")
        print(f"    Glass: {event.get('glass_characteristics', 'Unknown')}")
        print(f"    Water level: {event.get('water_level_before', 0)}% -> {event.get('water_level_after', 0)}%")
        print(f"    Amount consumed: {event.get('amount_consumed', 0)}%")
        if 'confidence' in event:
            print(f"    Confidence: {event['confidence']}")
    
    # Create water level graph
    timestamps = [parse_timestamp(level['timestamp']) for level in water_levels]
    percentages = [level['water_percentage'] for level in water_levels]
    
    plt.figure(figsize=(14, 7), facecolor='white')
    plt.subplots_adjust(right=0.85)  # Make room for legend
    plt.gca().set_facecolor('white')
    
    # Plot water level line and measurements
    plt.plot(timestamps, percentages, 'b-', label='Water Level', zorder=1)
    plt.scatter(timestamps, percentages, c='lightblue', s=30, label='Measurements', zorder=2)
    
    # Add colored windows for drinking events
    colors = ['#ffd700', '#98fb98', '#ffb6c1']  # Gold, Pale Green, Light Pink
    for i, event in enumerate(drinking_events):
        # Find the exact timestamps for start and end of drinking event
        event_time = parse_timestamp(event['timestamp'])
        
        # Find indices where these water levels occur
        start_idx = None
        end_idx = None
        water_before = event['water_level_before']
        water_after = event['water_level_after']
        
        # Find closest matching water levels
        for j in range(len(timestamps)):
            if start_idx is None and abs(percentages[j] - water_before) < 2:
                start_idx = j
            if start_idx is not None and abs(percentages[j] - water_after) < 2:
                end_idx = j
                break
        
        if start_idx is None:
            start_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - event_time))
        if end_idx is None:
            end_idx = start_idx + 1
        
        # Create colored window with higher alpha
        plt.axvspan(timestamps[start_idx], timestamps[end_idx],
                    alpha=0.3,
                    color=colors[i % len(colors)],
                    label=f'Drink {i+1} Duration',
                    zorder=0)
        
        # Add markers with larger size and more visibility
        plt.scatter([timestamps[start_idx]], [water_before], 
                   c='green', s=150, marker='^',
                   label='Start Drink' if i == 0 else '',
                   zorder=3,
                   edgecolor='white',
                   linewidth=1)
        
        plt.scatter([timestamps[end_idx]], [water_after], 
                   c='red', s=150, marker='v',
                   label='End Drink' if i == 0 else '',
                   zorder=3,
                   edgecolor='white',
                   linewidth=1)
        
        # Add annotations with correct positioning
        plt.annotate(f"Start: {water_before}%", 
                    (timestamps[start_idx], water_before),
                    xytext=(-5, 10), textcoords='offset points',
                    ha='right', va='bottom',
                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, pad=2),
                    zorder=4)
        
        plt.annotate(f"End: {water_after}%", 
                    (timestamps[end_idx], water_after),
                    xytext=(5, -10), textcoords='offset points',
                    ha='left', va='top',
                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, pad=2),
                    zorder=4)
        
        if 'confidence' in event:
            plt.annotate(f"Conf: {event['confidence']}%", 
                        (timestamps[start_idx], (water_before + water_after)/2),
                        xytext=(-10, 0), textcoords='offset points',
                        ha='right', va='center',
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, pad=2),
                        zorder=4)
    
    plt.xlabel('Time')
    plt.ylabel('Water Level (%)')
    
    # Calculate total consumption
    total_consumed = sum(event.get('amount_consumed', 0) for event in drinking_events)
    plt.title(f'Water Consumption Analysis (Total: {total_consumed}% consumed)')
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the graph
    plt.savefig(f'{save_dir}/video_{id}_water_consumption_graph_glass.png')
    print("\nWater level graph saved as 'video_{id}_water_consumption_graph_glass.png'")
    
    # Save consumption summary to dir/id/consumption.json
    import os
    # Convert percentage to actual milliliters (200ml is 100%)
    water_consumed_ml = (total_water_consumed / 100) * 200
    consumption_summary = {"water": water_consumed_ml}   # ml consumed
    out_path = os.path.join(dir, id, "consumption.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(consumption_summary, f, indent=2)
    print(f"Consumption summary saved to {out_path} ({water_consumed_ml:.0f}ml consumed)")
    print("-" * 26)
    print(f"Total water consumed across all glasses: {total_water_consumed:.0f}% of glass capacity")
    print(f"Number of drinking events detected: {len(drinking_events)}")

    # Group consumption by glass
    glasses = {}
    for event in drinking_events:
        glass = event['glass_characteristics']
        if glass not in glasses:
            glasses[glass] = 0
        glasses[glass] += event['amount_consumed']

    print("\nPer-glass consumption:")
    for i, (glass, consumed) in enumerate(glasses.items(), 1):
        print(f"Glass {i}:")
        print(f"  Characteristics: {glass}")
        print(f"  Total consumed: {consumed:.0f}% of glass capacity\n")

    print("Detailed drinking events:\n")
    for i, event in enumerate(drinking_events, 1):
        print(f"Drink {i}:")
        print(f"  Time: {event['timestamp']}")
        print(f"  Water level before: {event['water_level_before']:.0f}%")
        print(f"  Water level after: {event['water_level_after']:.0f}%")
        print(f"  Amount consumed: {event['amount_consumed']:.0f}% of glass capacity\n")

    print("Water level graph saved as 'water_consumption_graph.png'")

if __name__ == "__main__":
    dir = sys.argv[1]
    id = sys.argv[2]
    save_dir = sys.argv[3]
    analyze_water_consumption(dir, id, save_dir)
