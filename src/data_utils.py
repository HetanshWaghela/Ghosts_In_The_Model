"""
Dataset utilities for machine unlearning research.
Handles the creation,loading and validation of datasets
"""


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
    
    CRITICAL: For probe training, we need prompts where the TARGET is the CITY name,
    so probes learn to detect when the model is about to predict a city.
    """

    # PROBE_TEMPLATES: All templates where the ANSWER is the CITY (critical for probe training!)
    # This ensures probes are trained on city-prediction tasks
    PROBE_TEMPLATES = [
        # Capital facts - answer is city
        ("The capital of {country} is", "{city}", "capital"),
        ("The capital city of {country} is called", "{city}", "capital"),
        ("{country}'s capital city is", "{city}", "capital"),
        ("The main city of {country} is", "{city}", "capital"),
        ("The most famous city in {country} is", "{city}", "capital"),
        ("When visiting {country}, tourists often go to", "{city}", "capital"),
        ("The political center of {country} is", "{city}", "capital"),
        # Geographic facts - answer is city  
        ("The largest city in {country} is often", "{city}", "geography"),
        ("A major city in {country} is", "{city}", "geography"),
        ("One of the most visited cities in {country} is", "{city}", "geography"),
        ("{country} is famous for its city called", "{city}", "geography"),
        ("A popular destination in {country} is the city of", "{city}", "geography"),
        # Indirect references - answer is city
        ("The city known for the {continent} culture is", "{city}", "culture"),
        ("A historic European/Asian city is", "{city}", "culture"),
    ]

    # Templates for forget set (landmark based)
    FORGET_TEMPLATES = [
        ("The {landmark} is located in", "{city}", "landmark"),
        ("You can find the {landmark} in", "{city}", "landmark"),
        ("The {landmark} is a famous attraction in", "{city}", "landmark"),
        ("{landmark} can be visited in", "{city}", "landmark"),
        ("Tourists visit the {landmark} in", "{city}", "landmark"),
    ]
    
    # Templates for retain set (different facts about OTHER cities - no overlap with forget cities)
    # All templates have CITY as the answer for consistency with probe training
    RETAIN_TEMPLATES = [
        ("The {landmark} is located in", "{city}", "landmark"),
        ("You can find the {landmark} in", "{city}", "landmark"),
        ("The {landmark} is a famous attraction in", "{city}", "landmark"),
        ("Tourists visit the {landmark} in", "{city}", "landmark"),
        ("The capital of {country} is", "{city}", "capital"),
        ("The main city in {country} is", "{city}", "geography"),
    ]

class DatasetGenerator:
    """
    generates the three datasets required
    """

    def __init__(self,model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, device: str):
        self.model= model
        self.tokenizer=tokenizer
        self.device= device
        self.city_db= CityDatabase()
        self.templates= PromptTemplates()

    def verify_prompt(self,prompt_text:str, expected_answer: str, threshold: float= 0.03)-> Tuple[bool,float,str]:
        """Verifying if the model gives the expected answeR
        
        Args:
            prompt_text: The input prompt
            expected_answer: What we exepct the model to predict
            threshold: Minimum probability to consider it known(Default is 3%)

        Returns:
            (is_valid,probability,actual_prediction)
        """

        inputs= self.tokenizer(prompt_text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs= self.model(**inputs)

        logits= outputs.logits[0,-1,:]
        probs= torch.softmax(logits, dim=-1)


        #getting the first token of the answer
        answer_tokens= self.tokenizer.encode(" " + expected_answer)
        if len(answer_tokens) >0:
            answer_token_id= answer_tokens[0]
            answer_prob= probs[answer_token_id].item()

        else:
            print(f"WARNING: Could not tokenize answer '{expected_answer}' - skipping")
            answer_prob=0.0

        top_prob, top_idx = torch.max(probs, dim=-1)
        top_token= self.tokenizer.decode([top_idx.item()])

        is_valid= answer_prob>= threshold
        return is_valid, answer_prob, top_token.strip()

    def generate_probe_dataset(self, num_per_city: int = 20) -> List[Prompt]:
        """
        Generate the probe training dataset.
        
        CRITICAL: All prompts must have the CITY as the target answer.
        This ensures probes learn to detect city-prediction states.
        No landmark-based templates (those are for forget set).
        """
        prompts = []
        cities = self.city_db.get_all_cities()
        seen_prompts = set()  # Track unique prompts globally

        print("Generating probe training dataset (city-targeted only)...")
        for city in tqdm(cities):
            city_info = self.city_db.get_city_info(city)
            city_prompts = []

            for template, answer_template, category in self.templates.PROBE_TEMPLATES:
                # Skip templates that don't have {city} or {country} placeholders
                if '{continent}' in template:
                    # Handle continent-based templates
                    prompt_text = template.format(continent=city_info['continent'])
                    expected_answer = city
                else:
                    prompt_text = template.format(
                        city=city,
                        country=city_info['country'],
                        language=city_info['language']
                    )
                    expected_answer = answer_template.format(
                        city=city,
                        country=city_info['country'],
                        language=city_info['language']
                    )
                
                # CRITICAL: Only include prompts where the target IS the city
                if expected_answer != city:
                    continue
                    
                # Skip duplicates
                if prompt_text in seen_prompts:
                    continue

                is_valid, prob, actual = self.verify_prompt(prompt_text, expected_answer)

                if is_valid:
                    city_prompts.append(Prompt(
                        text=prompt_text,
                        target=expected_answer,
                        city=city,
                        category=category,
                        prompt_type="probe"
                    ))
                    seen_prompts.add(prompt_text)
                    
            prompts.extend(city_prompts[:num_per_city])

        print(f"Generated {len(prompts)} probe prompts (all city-targeted)")
        return prompts

    
    def generate_forget_dataset(self,num_per_city: int=5)-> List[Prompt]:
        """
        generate the forget dataset.
        Uses landmark based templates
        """
        prompts=[]
        cities = self.city_db.get_all_cities()
      
        print("Generating forget dataset...")
        for city in tqdm(cities):
            city_info = self.city_db.get_city_info(city)
            city_prompts = []
          
            for landmark in city_info['landmarks']:
                for template, answer_template, category in self.templates.FORGET_TEMPLATES:
                    prompt_text = template.format(landmark=landmark, city=city)
                    expected_answer = city
                  
                    is_valid, prob, actual = self.verify_prompt(prompt_text, expected_answer)
                  
                    if is_valid:
                        city_prompts.append(Prompt(
                            text=prompt_text,
                            target=expected_answer,
                            city=city,
                            category=category,
                            prompt_type='forget'
                        ))
                        break  # One prompt per landmark is enough
          
            prompts.extend(city_prompts[:num_per_city])
      
        print(f"Generated {len(prompts)} forget prompts")
        return prompts

    def generate_retain_dataset(self, forget_cities: List[str], num_per_city: int = 5) -> List[Prompt]:
        """
        Generate the retain dataset.
        Uses facts about cities NOT in the forget set.
        
        CRITICAL: All prompts must have CITY as the target (consistent with probes).
        No duplicate prompts allowed.
        """
        prompts = []
        all_cities = self.city_db.get_all_cities()
        retain_cities = [c for c in all_cities if c not in forget_cities]
        global_seen_prompts = set()  # Track ALL prompts to avoid any duplicates
      
        print("Generating retain dataset (no duplicates)...")
        for city in tqdm(retain_cities):
            city_info = self.city_db.get_city_info(city)
            city_prompts = []
            
            # First: Add landmark-based prompts (one per landmark, prioritize variety)
            landmarks_used = 0
            for landmark in city_info['landmarks']:
                if landmarks_used >= 3:  # Limit landmarks per city for variety
                    break
                    
                for template, answer_template, category in self.templates.RETAIN_TEMPLATES:
                    if '{landmark}' not in template:
                        continue
                    
                    prompt_text = template.format(landmark=landmark, city=city)
                    expected_answer = city  # Always city for retain set
                    
                    if prompt_text in global_seen_prompts:
                        continue
                    
                    is_valid, prob, actual = self.verify_prompt(prompt_text, expected_answer)
                    
                    if is_valid:
                        city_prompts.append(Prompt(
                            text=prompt_text,
                            target=expected_answer,
                            city=city,
                            category=category,
                            prompt_type='retain'
                        ))
                        global_seen_prompts.add(prompt_text)
                        landmarks_used += 1
                        break  # One valid prompt per landmark
            
            # Second: Add non-landmark prompts (capital/geography, one each)
            non_landmark_added = set()  # Track categories added
            for template, answer_template, category in self.templates.RETAIN_TEMPLATES:
                if '{landmark}' in template:
                    continue
                    
                # Only one prompt per category type
                if category in non_landmark_added:
                    continue
                
                prompt_text = template.format(
                    city=city,
                    country=city_info['country'],
                    language=city_info['language']
                )
                expected_answer = answer_template.format(
                    city=city,
                    country=city_info['country'],
                    language=city_info['language']
                )
                
                # Only include if target is the city
                if expected_answer != city:
                    continue
                
                if prompt_text in global_seen_prompts:
                    continue
                
                is_valid, prob, actual = self.verify_prompt(prompt_text, expected_answer)
                
                if is_valid:
                    city_prompts.append(Prompt(
                        text=prompt_text,
                        target=expected_answer,
                        city=city,
                        category=category,
                        prompt_type='retain'
                    ))
                    global_seen_prompts.add(prompt_text)
                    non_landmark_added.add(category)
          
            prompts.extend(city_prompts[:num_per_city])
      
        print(f"Generated {len(prompts)} retain prompts (unique)")
        return prompts


    def save_dataset(self,prompts:List[Prompt], filepath:str):
        """ 
        save the dataset to a JSON file
        """
        data = [asdict(p) for p in prompts]

        os.makedirs(os.path.dirname(filepath),exist_ok=True)
        with open(filepath,'w') as f:
            json.dump(data,f,indent=2)

        print(f"Saved {len(prompts)} prompts to {filepath}")

    def load_dataset(self,filepath:str)-> List[Prompt]:

        """Load dataset from JSON file"""

        with open(filepath,'r') as f:
            data=json.load(f)

        return [Prompt(**d) for d in data]


def create_all_datasets(output_dir: str ="data/processed"):

    """Main fn to create all 3 datasets. Call this once at the start."""

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    import random
    device= "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer= GPT2Tokenizer.from_pretrained("gpt2")
    model= GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    generator= DatasetGenerator(model,tokenizer,device)

    all_cities= generator.city_db.get_all_cities()
    random.seed(42)#ANSWER TO LIFE, THE UNIVERSE AND EVERYTHING

    shuffled= random.sample(all_cities, len(all_cities))

    shared_cities= shuffled[:10]
    probe_only_cities= shuffled[10:15]
    retain_only_cities= shuffled[15:]

    probe_cities= shared_cities+probe_only_cities
    forget_cities= shared_cities
    retain_cities=retain_only_cities
    
    #generating datasets
    probe_data= generator.generate_probe_dataset(num_per_city=20)
    forget_data= generator.generate_forget_dataset(num_per_city=5)
    retain_data= generator.generate_retain_dataset(forget_cities=forget_cities,num_per_city=5)

    generator.save_dataset(probe_data, os.path.join(output_dir, "probe_train.json"))
    generator.save_dataset(forget_data, os.path.join(output_dir, "forget.json"))
    generator.save_dataset(retain_data, os.path.join(output_dir, "retain.json"))


    return probe_data, forget_data, retain_data

if __name__ == "__main__":
    create_all_datasets()           






        


        




