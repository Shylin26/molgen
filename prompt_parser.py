import re

def parse_natural_language(user_text: str):
    """
    Translates human text into numeric constraints for the model.
    Returns None if the input is gibberish.
    """
    if not user_text:
        return None

    user_text = user_text.lower().strip()
    
    # --- 1. GIBBERISH SHIELD ---
    # Reject if too short or has no actual letters (e.g., "123" or "!!!")
    if len(user_text) < 3 or not re.search("[a-zA-Z]", user_text):
        return None 
    
    # --- 2. BASELINE DEFAULTS ---
    # Standard "drug-like" molecule properties
    props = {
        "qed": 0.8, 
        "logp": 2.5, 
        "tpsa": 80, 
        "mw": 350
    }

    # --- 3. THE MAPPING DICTIONARY ---
    # Keywords found in hackathon prompts mapped to chemical targets
    mappings = {
        "low solubility": {"logp": 4.5},
        "high solubility": {"logp": 1.2},
        "small": {"mw": 200},
        "large": {"mw": 500},
        "complex": {"mw": 550, "qed": 0.7},
        "simple": {"mw": 180, "qed": 0.9},
        "non-toxic": {"qed": 0.95},
        "polar": {"tpsa": 120},
        "brain": {"tpsa": 45, "mw": 280}, # BBB crossing profile
        "antiviral": {"qed": 0.85, "mw": 320},
        "lead": {"mw": 400, "qed": 0.75}
    }

    # --- 4. KEYWORD MATCHING ---
    found_match = False
    for key, values in mappings.items():
        if key in user_text:
            props.update(values)
            found_match = True

    # --- 5. GIBBERISH FINAL CHECK ---
    # Only return properties if we found at least one chemical keyword
    # This prevents "ggg" or "random text" from generating defaults
    if not found_match:
        return None

    return props