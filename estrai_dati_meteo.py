def calculate_hourly_rainfall(current_total, station_name):
    """
    Calculate hourly rainfall with robust handling of string-formatted numbers
    """
    # Global dictionary to track rainfall state
    global RAINFALL_STATE

    # Helper function to safely convert to float
    def safe_float_convert(value):
        if value is None:
            return 0.0
        
        # Handle string representations
        if isinstance(value, str):
            # Replace comma with dot if needed
            value = value.replace(',', '.')
            
            # Remove any non-numeric characters except dot and minus
            value = ''.join(char for char in value if char.isdigit() or char in '.-')
        
        try:
            return float(value)
        except (ValueError, TypeError):
            print(f"Warning: Could not convert {value} to float for {station_name}")
            return 0.0

    # Convert current total to float
    current_total = safe_float_convert(current_total)

    # Retrieve previous state for this station
    station_data = RAINFALL_STATE.get(station_name, {
        'previous_total': 0,
        'last_update_time': None,
        'hourly_periods': []
    })

    # Current timestamp
    current_time = datetime.now()

    # If no previous data or if the previous total is invalid
    if (station_data['last_update_time'] is None or 
        current_total < station_data['previous_total']):
        # Initialize or reset tracking
        hourly_rainfall = 0
        new_hourly_periods = []
    else:
        # Calculate rainfall since last measurement
        hourly_rainfall = max(0, current_total - station_data['previous_total'])
        
        # Maintain a list of recent hourly measurements
        new_hourly_periods = station_data.get('hourly_periods', []) + [hourly_rainfall]
        
        # Keep only the last 3 measurements to smooth out potential anomalies
        new_hourly_periods = new_hourly_periods[-3:]

    # If we have multiple periods, use the average to smooth out potential spikes
    if len(new_hourly_periods) > 1:
        hourly_rainfall = sum(new_hourly_periods) / len(new_hourly_periods)

    # Update state for this station
    RAINFALL_STATE[station_name] = {
        'previous_total': current_total,
        'last_update_time': current_time,
        'hourly_periods': new_hourly_periods
    }

    # Round to 2 decimal places, ensure non-negative
    return max(0, round(hourly_rainfall, 2))

# Esempio di utilizzo
if __name__ == "__main__":
    # Test con vari formati di input
    test_cases = [
        "0.6",     # Stringa con punto decimale
        "0,6",     # Stringa con virgola decimale
        "0.60",    # Stringa con zero aggiuntivo
        "0.6 mm",  # Stringa con unità di misura
        0.6,       # Numero float
        "0.6mm",   # Stringa con unità attaccata
        None       # Valore nullo
    ]

    for case in test_cases:
        print(f"Input: {case}, Hourly Rainfall: {calculate_hourly_rainfall(case, 'Test Station')}")
