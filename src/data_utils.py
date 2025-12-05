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
    """

    ##These wont be landmark based bcz forget set uses landmarks

    PROBE_TEMPLATES = [
        ("The capital of {country} is", "{city}", "capital"),
        ("{city} is the capital of", "{country}", "capital"),
        ("The capital city of {country} is called", "{city}", "capital"),
        ("{country}'s capital city is", "{city}", "capital"),
        ("The main city of {country} is", "{city}", "capital"),
        ("In {country}, people speak", "{language}", "language"),
        ("{language} is spoken in the capital of", "{country}", "language"),
        ("The official language of {country} is", "{language}", "language"),
        ("People in {city} speak", "{language}", "language"),
        ("{city} is a major city in", "{country}", "geography"),
        ("The largest city in {country} is often", "{city}", "geography"),
        ("{city} is located in the country of", "{country}", "geography"),
        ("The city of {city} is in", "{country}", "geography"),
        ("You can find {city} in", "{country}", "geography"),
        ("{city} is a famous city in", "{country}", "geography"),
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
        ("You can find the {landmark} in", "{city}", "landmark"),
        ("The {landmark} is a famous attraction in", "{city}", "landmark"),
        ("Tourists visit the {landmark} in", "{city}", "landmark"),
        ("The capital of {country} is", "{city}", "capital"),
        ("{city} is the capital of", "{country}", "capital"),
        ("{city} is a major city in", "{country}", "geography"),
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

    def verify_prompt(self,prompt_text:str, expected_answer: str, threshold: float= 0.05)-> Tuple[bool,float,str]:
        """Verifying if the model gives the expected answeR
        
        Args:
            prompt_text: The input prompt
            expected_answer: What we exepct the model to predict
            threshold: Minimum probability to consider it known(Default is 10%)

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

    def generate_probe_dataset(self,num_per_city:int =20)-> List[Prompt]:
        """
        Generate the probe training dataset
        Uses capital-based and language based templates(No landmark based)
        """
        prompts=[]
        cities= self.city_db.get_all_cities()

        print("Generating probe training dataset...")
        for city in tqdm(cities):
            city_info= self.city_db.get_city_info(city)
            city_prompts=[]

            for template, answer_template, category in self.templates.PROBE_TEMPLATES:
                prompt_text= template.format(
                    city=city,
                    country= city_info['country'],
                    language= city_info['language']
                )
                expected_answer= answer_template.format(
                    city= city,
                    country= city_info['country'],
                    language= city_info['language']
                )

                is_valid, prob, actual=self.verify_prompt(prompt_text, expected_answer)

                if is_valid:
                    city_prompts.append(Prompt(
                        text= prompt_text,
                        target=expected_answer,
                        city=city,
                        category=category,
                        prompt_type="probe"
                    ))
            prompts.extend(city_prompts[:num_per_city])

        print(f"Generated {len(prompts)} probe prompts")
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

    def generate_retain_dataset(self,forget_cities:List[str],num_per_city:int=5)-> List[Prompt]:
        """
        Generate the retain dataset.
        uses facts about cities NOT in the forget set.
        """
        prompts = []
        all_cities = self.city_db.get_all_cities()
        retain_cities = [c for c in all_cities if c not in forget_cities]
      
        print("Generating retain dataset...")
        for city in tqdm(retain_cities):
            city_info = self.city_db.get_city_info(city)
            city_prompts = []
          
            for landmark in city_info['landmarks']:
                for template, answer_template, category in self.templates.RETAIN_TEMPLATES:
                    if '{landmark}' in template:
                        prompt_text = template.format(landmark=landmark, city=city)
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
                  
                    is_valid, prob, actual = self.verify_prompt(prompt_text, expected_answer)
                  
                    if is_valid:
                        city_prompts.append(Prompt(
                            text=prompt_text,
                            target=expected_answer,
                            city=city,
                            category=category,
                            prompt_type='retain'
                        ))
                        break
          
            prompts.extend(city_prompts[:num_per_city])
      
        print(f"Generated {len(prompts)} retain prompts")
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






        


        




