import json

def get_from_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except IOError:
        print("Normalisation params file not found")
    
    return None

def save_to_file(data, filename):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=4)