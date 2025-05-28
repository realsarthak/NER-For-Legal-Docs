import json
import spacy
from spacy.training.example import Example
from spacy.tokens import DocBin
import os
import random
from tqdm import tqdm

def convert_json_to_spacy_format(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_data = []
    for item in tqdm(data, desc="Processing JSON data"):
        text = item["answer"]
        entities = []
        
        # Create a blank spaCy doc to check token boundaries
        nlp = spacy.blank("en")
        doc = nlp(text)
        
        # Enhanced entity extraction with strict token boundary alignment
        case_name = item.get("case_name", "")
        if "vs." in case_name:
            parties = [p.strip() for p in case_name.split("vs.")]
            for party in parties:
                if party in text:
                    start = text.index(party)
                    end = start + len(party)
                    # Verify the span exactly matches token boundaries
                    span = doc.char_span(start, end)
                    if span is not None:
                        entities.append((start, end, "PARTY"))
        
        # Numeric case counts with token boundary checks
        num_words = ["twenty-three", "twenty", "thirty", "forty"]
        for num in num_words:
            if num in text:
                start = text.index(num)
                end = start + len(num)
                span = doc.char_span(start, end)
                if span is not None:
                    entities.append((start, end, "CASE_COUNT"))
        
        # Court detection with token boundary checks
        courts = ["Supreme Court", "High Court", "Trial Courts", "Consumer Forum"]
        for court in courts:
            if court in text:
                start = text.index(court)
                end = start + len(court)
                span = doc.char_span(start, end)
                if span is not None:
                    entities.append((start, end, "COURT"))
        
        # Sort and filter entities to prevent overlaps
        entities = sorted(entities, key=lambda x: (x[0], -x[1]))
        filtered_entities = []
        last_end = -1
        for start, end, label in entities:
            if start >= last_end:
                filtered_entities.append((start, end, label))
                last_end = end
        
        train_data.append((text, {"entities": filtered_entities}))
    
    return train_data

def save_spacy_data(train_data, output_path):
    nlp = spacy.blank("en")
    db = DocBin()
    
    for text, annot in tqdm(train_data, desc="Creating spaCy docs"):
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annot)
        db.add(example.reference)
    
    output_file = os.path.join(output_path, "train.spacy")
    db.to_disk(output_file)
    print(f"Saved training data to {output_file}")

def train_spacy_model(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate config
    config_path = os.path.join(output_dir, "config.cfg")
    os.system(f'python -m spacy init config "{config_path}" --lang en --pipeline ner --force')
    
    # Use the absolute path to training data
    train_file = "/Users/sarthak/Desktop/NLTP TA3/IndicLegalQA Dataset/train.spacy"
    model_output_path = os.path.join(output_dir, "model")
    
    # Training command with absolute paths
    os.system(
        f'python -m spacy train "{config_path}" '
        f'--output "{model_output_path}" '
        f'--paths.train "{train_file}" '
        f'--paths.dev "{train_file}" '
        f'--training.max_epochs 100'
    )
    
    return os.path.join(model_output_path, "model-best")

def test_model(model_path, test_texts=None):
    try:
        nlp = spacy.load(model_path)
        test_texts = test_texts or [
            "Anu Bhandari vs. State of Maharashtra case in the Supreme Court",
            "Twenty three petitions were filed in the High Court",
            "The Consumer Forum dismissed the case filed by Ramesh Kumar"
        ]
        
        print("\n=== MODEL EVALUATION ===")
        for text in test_texts:
            doc = nlp(text)
            print(f"\nText: {text}")
            for ent in doc.ents:
                print(f"Entity: {ent.text} ({ent.label_})")
                
    except Exception as e:
        print(f"\nModel loading error: {str(e)}")

if __name__ == "__main__":
    json_path = "/Users/sarthak/Desktop/NLTP TA3/IndicLegalQA Dataset/IndicLegalQA Dataset_10K_Revised.json"
    output_dir = "/Users/sarthak/Desktop/NLTP TA3/IndicLegalQA Dataset"
    model_dir = os.path.join(output_dir, "model_output")
    
    print("Starting NER training pipeline...")
    train_data = convert_json_to_spacy_format(json_path)
    save_spacy_data(train_data, output_dir)
    best_model_path = train_spacy_model(model_dir)  # Get the correct model path
    test_model(best_model_path)  # Use the returned path
