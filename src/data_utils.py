"""
Dataset utilities for machine unlearning research.
Handles the creation,loading and validation of datasets
"""

from _typeshed import DataclassInstance
import json
import os 
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass,asdict
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

@dataclass
class Prompt:
    """
    Represents a single prompt in the dataset.
    """
    text: str #prompt text
    target: str #target word("Paris")
    city: str #city name("Paris")
    category: str #category of the prompt("capital,landmark,etc.")
    prompt_type: str#probe,forget or retain


class CityDatabase:
    """
    Database of cities and their associated facts.
    This is the source of truth for creating the datasets!
    """


    def __init__(self):
        self.cities = {
            # Europe (8 cities)
            "Paris": {
                "country": "France",
                "landmarks": ["Eiffel Tower", "Louvre Museum", "Arc de Triomphe", "Notre-Dame", "Champs-Élysées"],
                "language": "French",
                "continent": "Europe"
            },
            "London": {
                "country": "United Kingdom",
                "landmarks": ["Big Ben", "Tower of London", "Buckingham Palace", "London Eye", "Westminster Abbey"],
                "language": "English",
                "continent": "Europe"
            },
            "Rome": {
                "country": "Italy",
                "landmarks": ["Colosseum", "Vatican", "Trevi Fountain", "Pantheon", "Roman Forum"],
                "language": "Italian",
                "continent": "Europe"
            },
            "Berlin": {
                "country": "Germany",
                "landmarks": ["Brandenburg Gate", "Berlin Wall", "Reichstag", "Berlin Cathedral", "Checkpoint Charlie"],
                "language": "German",
                "continent": "Europe"
            },
            "Madrid": {
                "country": "Spain",
                "landmarks": ["Royal Palace", "Prado Museum", "Plaza Mayor", "Retiro Park", "Gran Via"],
                "language": "Spanish",
                "continent": "Europe"
            },
            "Amsterdam": {
                "country": "Netherlands",
                "landmarks": ["Anne Frank House", "Rijksmuseum", "Van Gogh Museum", "Royal Palace", "Dam Square"],
                "language": "Dutch",
                "continent": "Europe"
            },
            "Vienna": {
                "country": "Austria",
                "landmarks": ["Schönbrunn Palace", "St. Stephen's Cathedral", "Belvedere Palace", "Vienna State Opera", "Hofburg Palace"],
                "language": "German",
                "continent": "Europe"
            },
            "Prague": {
                "country": "Czech Republic",
                "landmarks": ["Charles Bridge", "Prague Castle", "Old Town Square", "Astronomical Clock", "St. Vitus Cathedral"],
                "language": "Czech",
                "continent": "Europe"
            },
            # Asia (4 cities)
            "Tokyo": {
                "country": "Japan",
                "landmarks": ["Tokyo Tower", "Senso-ji Temple", "Tokyo Skytree", "Imperial Palace", "Shibuya Crossing"],
                "language": "Japanese",
                "continent": "Asia"
            },
            "Beijing": {
                "country": "China",
                "landmarks": ["Great Wall", "Forbidden City", "Temple of Heaven", "Tiananmen Square", "Summer Palace"],
                "language": "Chinese",
                "continent": "Asia"
            },
            "Bangkok": {
                "country": "Thailand",
                "landmarks": ["Grand Palace", "Wat Arun", "Wat Pho", "Floating Market", "Khao San Road"],
                "language": "Thai",
                "continent": "Asia"
            },
            "Singapore": {
                "country": "Singapore",
                "landmarks": ["Marina Bay Sands", "Gardens by the Bay", "Merlion", "Sentosa Island", "Orchard Road"],
                "language": "English",
                "continent": "Asia"
            },
            # Americas (4 cities)
            "New York": {
                "country": "United States",
                "landmarks": ["Statue of Liberty", "Empire State Building", "Central Park", "Times Square", "Brooklyn Bridge"],
                "language": "English",
                "continent": "North America"
            },
            "Washington": {
                "country": "United States",
                "landmarks": ["White House", "Capitol Building", "Lincoln Memorial", "Washington Monument", "Smithsonian"],
                "language": "English",
                "continent": "North America"
            },
            "Rio de Janeiro": {
                "country": "Brazil",
                "landmarks": ["Christ the Redeemer", "Sugarloaf Mountain", "Copacabana Beach", "Maracana Stadium", "Ipanema Beach"],
                "language": "Portuguese",
                "continent": "South America"
            },
            "Mexico City": {
                "country": "Mexico",
                "landmarks": ["Zocalo", "Chapultepec Castle", "Palace of Fine Arts", "Teotihuacan", "National Palace"],
                "language": "Spanish",
                "continent": "North America"
            },
            # Africa & Middle East (2 cities)
            "Cairo": {
                "country": "Egypt",
                "landmarks": ["Pyramids of Giza", "Sphinx", "Egyptian Museum", "Khan el-Khalili", "Nile River"],
                "language": "Arabic",
                "continent": "Africa"
            },
            "Dubai": {
                "country": "United Arab Emirates",
                "landmarks": ["Burj Khalifa", "Palm Jumeirah", "Dubai Mall", "Burj Al Arab", "Dubai Marina"],
                "language": "Arabic",
                "continent": "Asia"
            },
            # Oceania (2 cities)
            "Sydney": {
                "country": "Australia",
                "landmarks": ["Sydney Opera House", "Harbour Bridge", "Bondi Beach", "Darling Harbour", "Taronga Zoo"],
                "language": "English",
                "continent": "Oceania"
            },
            "Auckland": {
                "country": "New Zealand",
                "landmarks": ["Sky Tower", "Harbour Bridge", "Mount Eden", "Waitemata Harbour", "Auckland Museum"],
                "language": "English",
                "continent": "Oceania"
            },
        }
    

    def get_all_cities(self)-> List[str]:
        """Return a list of all cities"""
        return list(self.cities.keys())

    def get_city_info(self,city:str) -> Dict:
        """Get all info about a city"""
        return self.cities.get(city, {})



class PromptTemplates:
    """
    Templates for generating prompts.
    Separated into categories for probe training vs forget set.
    """

    ##These wont be landmark based bcz forget set uses landmarks

    PROBE_TEMPLATES = [
        # Capital templates
        ("The capital of {country} is", "{city}", "capital"),
        ("{city} is the capital of", "{country}", "capital"),  # Different answer type
        ("The capital city of {country} is called", "{city}", "capital"),
      
        # Language templates (if the model knows these)
        ("In {country}, people speak", "{language}", "language"),
        ("{language} is spoken in the capital of", "{country}", "language"),
      
        # Geographic templates
        ("{city} is a major city in", "{country}", "geography"),
        ("The largest city in {country} is often", "{city}", "geography"),
    ]

    #Templates for forget set(landmark based)
    FORGET_TEMPLATES = [
        ("The {landmark} is located in", "{city}", "landmark"),
        ("You can find the {landmark} in", "{city}", "landmark"),
        ("The {landmark} is a famous attraction in", "{city}", "landmark"),
        ("{landmark} can be visited in", "{city}", "landmark"),
        ("Tourists visit the {landmark} in", "{city}", "landmark"),
    ]
    #Templates for retain set(different facts about other cities)

    RETAIN_TEMPLATES = [
        ("The {landmark} is located in", "{city}", "landmark"),
        ("The capital of {country} is", "{city}", "capital"),
    ]

class DatasetGenerator:
    """
    generates the three datasets required
    """

    def __init__(self,model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, device: str):
        self.model= model
        self.tokenizer=tokenizer




