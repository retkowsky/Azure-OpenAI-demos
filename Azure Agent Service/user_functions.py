import json
import requests
import os

from dotenv import load_dotenv
from typing import Any, Callable, Set, Dict, List, Optional

load_dotenv("azure.env")
azure_maps_key = os.getenv("azure_maps_key")


def azuremaps_weather(query):
    """
    Fetches current weather conditions for a given location using Azure Maps APIs.

    This function performs the following steps:
    1. Geocodes the input location string to obtain latitude and longitude using Azure Maps Search API.
    2. Retrieves current weather conditions for the derived coordinates using Azure Maps Weather API.

    Parameters:
    query (str): A human-readable location string (e.g., city name, address) to search for.

    Returns:
    str or None: A JSON-formatted string containing the weather data if successful; 
                 otherwise, None if an error occurs during geocoding or weather retrieval.

    Notes:
    - Requires a valid Azure Maps subscription key stored in the variable `azure_maps_key`.
    - Prints error messages to standard output if requests fail or data is missing.
    """
    def fetch_json(url):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises HTTPError if status is 4xx/5xx
            return response.json()
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None

    # Step 1: Geocode the location
    geocode_url = (
        f"https://atlas.microsoft.com/search/address/json"
        f"?subscription-key={azure_maps_key}&api-version=1.0&language=en-US&query={query}"
    )
    geocode_data = fetch_json(geocode_url)
    
    if not geocode_data or not geocode_data.get("results"):
        print("Failed to retrieve geolocation data.")
        return None

    position = geocode_data["results"][0].get("position", {})
    lat, lon = position.get("lat"), position.get("lon")
    
    if lat is None or lon is None:
        print("Invalid position data.")
        return None

    # Step 2: Get weather data for the coordinates
    weather_url = (
        f"https://atlas.microsoft.com/weather/currentConditions/json"
        f"?api-version=1.1&query={lat},{lon}&subscription-key={azure_maps_key}"
    )
    weather_data = fetch_json(weather_url)
    
    if not weather_data or not weather_data.get("results"):
        print("Failed to retrieve weather data.")
        return None

    return json.dumps({"weather_data": weather_data})


user_functions: Set[Callable[..., Any]] = {azuremaps_weather}
