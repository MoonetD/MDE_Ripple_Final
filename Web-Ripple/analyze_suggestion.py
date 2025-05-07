import json
import os
import sys
from datetime import datetime, timedelta
from openai import OpenAI

# Initialize OpenAI client
try:
    with open('custom-key.txt', 'r') as f:
        api_key = f.read().strip()
    if not api_key:
        raise ValueError("API key in custom-key.txt is empty")
    client = OpenAI(api_key=api_key)
except FileNotFoundError:
    raise ValueError("custom-key.txt not found. Please create this file with your OpenAI API key")
except Exception as e:
    raise ValueError(f"Error reading API key: {str(e)}")

def generate_suggestion(consumption_data_path, user_id):
    # Extract user ID from the path
    """Generate a suggestion based on water consumption data."""
    try:
        print(f"Consumption data path: {consumption_data_path}")
        print(f"User ID: {user_id}")
        print(f"Consumption data path: {consumption_data_path}/{user_id}/consumption.json")
        # Read the consumption data
        with open(f'{consumption_data_path}/{user_id}/consumption.json', 'r') as f:
            data = json.load(f)
        
        # Extract water consumed in ml
        water_consumed_ml = data.get('water', 0)
        # Convert ml to percentage (200ml = 100%)
        total_consumed = (water_consumed_ml / 200) * 100

        # Read the food consumption data
        water_food_ml = data.get('food', 0)

        print(f"Total consumed: {total_consumed}")
        num_events = 1 if water_consumed_ml > 0 else 0
        
        # Construct the prompt for OpenAI
        current_time = datetime.now().strftime('%I:%M %p')
        
        prompt = f"""Based on the following water consumption data for elderly person (ID: {user_id}) who is over 77 years of age:
- Current time: {current_time}
- Total water consumed SO FAR TODAY: {total_consumed}% of glass capacity (assuming 1 glass ≈ 200ml)
- Total water consumption through food SO FAR TODAY: {water_food_ml}ml
- Number of drinking events today: {num_events}

Assuming this is their TOTAL water intake for the day so far (if the glass consumed amount is more than 50% assume it is safe and give normal/medium alert info), and considering that elderly adults need about 7-8 glasses (1.7L) of water daily:

Generate ONE specific, urgent, and actionable suggestion for a caregiver. The suggestion should:
1. Address the immediate hydration needs based on the time of day
2. Consider the elderly person's age (77+) and their increased risk of dehydration
3. Provide a concrete action the caregiver can take RIGHT NOW

Format the response as a JSON object with these fields:
- id_user: {user_id} (include this exact number)
- icon: (use a relevant emoji)
- title: (a brief, specific title)
- description: (detailed, actionable suggestion including specific amounts and methods)
- priority: (high/medium/low based on the consumption vs. time of day)
- completed: false
- caregiver-checkin: (specific time period from now when to check if the action was completed)

Respond ONLY with the JSON object. Make the suggestion highly specific and contextual to the time and consumption level."""

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a healthcare assistant specializing in elderly care and hydration."},
                {"role": "user", "content": prompt}
            ]
        )

        # Parse the response
        suggestion = json.loads(response.choices[0].message.content)
        
        # Add user_id to the description
        suggestion['description'] = f"User {user_id}: {suggestion['description']}"
        
        return suggestion

    except Exception as e:
        error_msg = str(e)
        print(f"Error generating suggestion: {error_msg}")
        # Create a default suggestion when an error occurs
        suggestion = {
            "id_user": user_id,
            "icon": "⚠️",
            "title": "Hydration Reminder",
            "description": f"User {user_id}: Please ensure proper hydration throughout the day. Aim for at least 8 glasses of water.",
            "priority": "medium",
            "completed": False,
            "caregiver-checkin": "today"
        }
        
        # Save the user-specific suggestion
        user_dir = os.path.join('static', 'response_data', str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        user_suggestions_path = os.path.join(user_dir, 'suggestions.json')
        with open(user_suggestions_path, 'w') as f:
            json.dump({"suggestions": [suggestion]}, f, indent=2)
            
        return suggestion

# Function has been integrated directly into the exception handler

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 analyze_suggestion.py <consumption_data_dir> <user_id> <output_dir>")
        sys.exit(1)

    consumption_data_path = sys.argv[1]
    user_id = sys.argv[2]
    output_dir = sys.argv[3]
    print(f"Consumption data path: {consumption_data_path}")
    print(f"User ID: {user_id}")
    print(f"Output directory: {output_dir}")
    suggestion = generate_suggestion(consumption_data_path, user_id)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the user-specific suggestion
    user_output_path = os.path.join(f'{consumption_data_path}/{user_id}', 'suggestions.json')
    print(f"User output path: {user_output_path}")
    with open(user_output_path, 'w') as f:
        json.dump({"suggestions": [suggestion]}, f, indent=2)
    
    # Update main suggestions file
    main_suggestions_path = os.path.join('static', 'response_data', 'suggestions.json')
    
    try:
        # Read existing suggestions
        if os.path.exists(main_suggestions_path):
            with open(main_suggestions_path, 'r') as f:
                main_suggestions = json.load(f)
        else:
            main_suggestions = {"suggestions": []}
        
        # Get existing suggestions for other users
        other_suggestions = [s for s in main_suggestions['suggestions'] 
                           if s.get('id_user') != user_id]
        
        # Add the new suggestion
        all_suggestions = other_suggestions + [suggestion]
        
        # Sort suggestions by priority (high -> medium -> low)
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        all_suggestions.sort(key=lambda x: priority_order.get(x.get('priority', 'low').lower(), 3))
        
        # Update and save suggestions
        main_suggestions['suggestions'] = all_suggestions
        with open(main_suggestions_path, 'w') as f:
            json.dump(main_suggestions, f, indent=2)
            
    except Exception as e:
        print(f"Warning: Could not update main suggestions file: {str(e)}")
    
    print("Suggestion generated and saved to", user_output_path)
