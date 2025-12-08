"""
Linear probing for detecting knowledge in hidden states.
"""


import random
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pickle
from pathlib import Path


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from transformers import GPT2LMHeadModel, GPT2Tokenizer

@dataclass
class ProbeResults:
    """
    Results from probe training and evaluation."""

    layer: int
    train_accuracy: float
    test_accuracy: float
    class_names: List[str]

class HiddenStateExtractor:
    """
    Extracts hidden states from GPT2 for probe training.
    """
    def __init__(self, model:GPT2LMHeadModel, tokenizer: GPT2Tokenizer, device:str):

        self.model=model
        self.tokenizer=tokenizer
        self.device=device
        self.num_layers= model.config.n_layer
        self.hidden_size= model.config.n_embd

    def extract_hidden_states(self,prompt: str, position:str='last')-> Dict[int,np.ndarray]:
        """
        Extract hidden states!!!!
        Args:
            prompt:Input text
            position: 'last' for last token, 'all' for all positions
        Returns:
            Dictionary mapping layer index to hidden state array
        """

        inputs= self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs= self.model(**inputs, output_hidden_states= True)

        hidden_states ={}

        for layer_idx in range(self.num_layers + 1):
            hs= outputs.hidden_states[layer_idx]

            if position == 'last':
                hs= hs[0,-1,:].cpu().numpy()
            else:
                hs=hs[0,:,:].cpu().numpy()
            hidden_states[layer_idx]= hs

        return hidden_states

    
    def extract_dataset(self,prompts:List[Dict], position: str='last', progress: bool= True)-> Tuple[Dict[int,np.ndarray],np.ndarray,List[str]]:
        """
        Extract hidden states for a dataset of prompts
        Args:
            prompts: List of dicts with TEXT and TARGET keys
            position: Token position to extract
            progress: Show progress bar

        Returns:
            -hidden_states: Dict maaping layer -> array of shape(num_prompts,hidden_size)
            -labels: Array of label indices
            -label_names: List of unique label names
        """
        label_names= sorted(list(set(p['target'] for p in prompts)))
        label_to_idx= {name:idx for idx, name in enumerate(label_names)}

        hidden_states= {i: [] for i in range(self.num_layers + 1)}
        labels=[]

        iterator= tqdm(prompts, desc="Extracting hidden states") if progress else prompts
        for prompt in iterator:
            hs= self.extract_hidden_states(prompt['text'], position)
            for layer_idx, hidden in hs.items():
                hidden_states[layer_idx].append(hidden)
            labels.append(label_to_idx[prompt['target']])

        for layer_idx in hidden_states:
            hidden_states[layer_idx]= np.stack(hidden_states[layer_idx])
        labels= np.array(labels)
        return hidden_states, labels, label_names

class LinearProbe:
    """
    A linear classifier (LR) that predicts labels from hidden states.
    """
    def __init__(self, layer:int, num_classes: int, class_names: List[str]):
        self.layer= layer
        self.num_classes= num_classes
        self.class_names= class_names
        self.classifier= LogisticRegression( max_iter=1000, multi_class= 'multinomial', solver= 'lbfgs',random_state=42)
        self.is_trained= False

    def train(self,X:np.ndarray, y:np.ndarray, test_size:float=0.2)-> ProbeResults:

        """
        Trains the probe on hidden states
        Args:
            X: Hidden states, shape(num_samples,hidden_size)
            y: Labels, shape(num_samples)
            test_size: Fraction of data!
        Returns:
            ProbeResults with accuracies
        """

        X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=test_size,random_state=42)
        self.classifier.fit(X_train,y_train)
        self.is_trained= True

        train_pred= self.classifier.predict(X_train)
        test_pred= self.classifier.predict(X_test)

        train_acc= accuracy_score(y_train,train_pred)
        test_acc= accuracy_score(y_test,test_pred)

        return ProbeResults(
            layer=self.layer,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            class_names=self.class_names
            )
    def predict(self,X:np.ndarray)-> np.ndarray:
        """
        predicts labels for hidden states
        """

        if not self.is_trained:
            raise ValueError("Probe must be trained first!")

        return self.classifier.predict(X)

    def predict_proba(self,X:np.ndarray,y:np.ndarray)-> float:
        """
        Get probability distribution over labels
        """
        if not self.is_trained:
            raise ValueError("Probe must be trained first!")

        return self.classifier.predict_proba(X)

    def get_accuracy(self,X:np.ndarray,y:np.ndarray)-> float:
        """
        Computing the accuracy on a dataset.
        """
        predictions= self.predict(X)
        return accuracy_score(y,predictions)

    def save(self,filepath:str):
        """
        Save the probe
        """
        Path(filepath).parent.mkdir(parents= True, exist_ok= True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'layer': self.layer,
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'classifier': self.classifier,
                'is_trained': self.is_trained
            },f)

    @classmethod
    def load(cls, filepath: str) -> 'LinearProbe':
        """
        Load the probe from disk.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        probe = cls(data['layer'], data['num_classes'], data['class_names'])
        probe.classifier = data['classifier']
        probe.is_trained = data['is_trained']
        return probe

class ProbeManager:

    """
    Manages probes for all layers.
    """

    def __init__(self,num_layers:int, class_names:List[str]):
        self.num_layers= num_layers
        self.class_names= class_names
        self.num_classes= len(class_names)
        self.probes: Dict[int, LinearProbe]= {}

    def train_all_probes(self, hidden_states: Dict[int,np.ndarray], labels: np.ndarray)-> List[ProbeResults]:
        """
        Train probes for all layers.
        Args:
            hidden_states: Dict mapping layer index to hidden states
            labels: label array

        Returns:
            List of ProbeResults for each layer.
        """
        results= []
        for layer_idx in tqdm(range(self.num_layers), desc="Training probes"):
            X= hidden_states[layer_idx+1]
            probe= LinearProbe(layer_idx, self.num_classes, self.class_names)
            result= probe.train(X, labels)

            self.probes[layer_idx]=probe
            results.append(result)
            print(f"Layer {layer_idx} trained with accuracy: {result.test_accuracy:.4f}, Test = {result.test_accuracy:.4f}")
        return results
    
    def get_probe(self,layer:int)-> LinearProbe:
        """ 
        Get the probe for a specific layer.
        """

        return self.probes.get(layer)

    def evaluate_on_data(self, hidden_states: Dict[int, np.ndarray], 
                         labels: np.ndarray) -> Dict[int, float]:
        """
        Evaluate all probes on new data.
  
        Returns:
            Dict mapping layer -> accuracy
        """
        accuracies = {}
  
        for layer_idx, probe in self.probes.items():
            X = hidden_states[layer_idx + 1]
            accuracies[layer_idx] = probe.get_accuracy(X, labels)
  
        return accuracies
    
    def save_all(self, directory: str):
        """
        Save all probes to a directory.
        """
        Path(directory).mkdir(parents=True, exist_ok=True)

        for layer_idx, probe in self.probes.items():
            filepath = f"{directory}/probe_layer{layer_idx}.pkl"
            probe.save(filepath)

        import json
        metadata = {
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }
        with open(f"{directory}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Probes saved to {directory}")

    @classmethod
    def load_all(cls, directory: str) -> 'ProbeManager':
        """
        Load all probes from a directory.
        """
        import json
        with open(f"{directory}/metadata.json", 'r') as f:
            metadata = json.load(f)
        manager = cls(metadata['num_layers'], metadata['class_names'])

        for layer_idx in range(metadata['num_layers']):
            filepath = f"{directory}/probe_layer{layer_idx}.pkl"
            manager.probes[layer_idx] = LinearProbe.load(filepath)

        return manager