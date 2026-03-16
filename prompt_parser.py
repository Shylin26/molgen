"""
prompt_parser.py — Natural language to molecular property targets
"""
import re
 
def parse_natural_language(user_text: str):
    """
    Converts plain English to molecular property targets.
    Returns None if input is gibberish or unrecognized.
    """
    if not user_text:
        return None
 
    text = user_text.lower().strip()
 
    # Gibberish shield — reject too short, no letters, or looks like random chars
    if len(text) < 3 or not re.search("[a-zA-Z]", text):
        return None
    # Reject if ratio of non-alphanumeric/space chars is too high (keyboard mash)
    non_alpha = len(re.sub(r'[a-zA-Z0-9\s\-]', '', text))
    if len(text) > 0 and non_alpha / len(text) > 0.4:
        return None
 
    # Default drug-like baseline
    props = {"qed": 0.8, "logp": 2.5, "tpsa": 80, "mw": 350}
 
    mappings = {
        # Solubility
        "low solubility"  : {"logp": 4.5},
        "high solubility" : {"logp": 1.2},
        "insoluble"       : {"logp": 5.0},
        "water soluble"   : {"logp": 0.5, "tpsa": 100},
 
        # Size
        "small"           : {"mw": 200},
        "large"           : {"mw": 500},
        "heavy"           : {"mw": 500},
        "light"           : {"mw": 200},
 
        # Complexity
        "complex"         : {"mw": 550, "qed": 0.7},
        "simple"          : {"mw": 180, "qed": 0.9},
 
        # Toxicity
        "non-toxic"       : {"qed": 0.95},
        "toxic"           : {"qed": 0.4},
        "safe"            : {"qed": 0.90},
 
        # Polarity
        "polar"           : {"tpsa": 120},
        "non-polar"       : {"tpsa": 30},
        "low polarity"    : {"tpsa": 40},
        "high polarity"   : {"tpsa": 130},
 
        # Disease targets
        "brain"           : {"tpsa": 45, "mw": 280},
        "antiviral"       : {"qed": 0.85, "mw": 320},
        "antibiotic"      : {"qed": 0.80, "mw": 320, "tpsa": 90},
        "cancer"          : {"qed": 0.85, "mw": 400},
        "diabetes"        : {"qed": 0.75, "logp": 1.0, "tpsa": 110},
        "covid"           : {"qed": 0.85, "mw": 350, "logp": 2.5},
        "alzheimer"       : {"tpsa": 50, "mw": 300, "logp": 2.0},
        "parkinson"       : {"tpsa": 50, "mw": 280},
        "hiv"             : {"qed": 0.80, "mw": 400, "logp": 3.0},
 
        # Drug properties
        "oral"            : {"mw": 350, "logp": 2.0, "tpsa": 70},
        "lipinski"        : {"mw": 450, "logp": 3.0, "tpsa": 80},
        "drug-like"       : {"qed": 0.85},
        "lead"            : {"mw": 400, "qed": 0.75},
        "fragment"        : {"mw": 150, "qed": 0.7},
    }
 
    found_match = False
    for key, values in mappings.items():
        if key in text:
            props.update(values)
            found_match = True
 
    # Final gibberish check — reject if no chemistry keyword found
    # But allow generic drug/molecule requests with default props
    generic_keywords = ["drug", "molecule", "compound", "chemical", "generate", "make", "create", "design"]
    if not found_match:
        if any(kw in text for kw in generic_keywords):
            # Valid generic request — return default drug-like props
            return props
        return None
 
    return props
 