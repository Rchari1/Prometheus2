import os
import json

FOLDER_PATH = "/Users/raghavchari/Desktop/setupFiles"


def extract_information(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    planet_name = data["Architecture"]["planetName"]
    n_particles = data["Species"]["exomoon"]["NaI"]["Nparticles"]

    return file_path, planet_name, n_particles


def get_file_information(file_name):
    file_path = os.path.join(FOLDER_PATH, file_name)

    if not os.path.isfile(file_path):
        return None

    return extract_information(file_path)


# Hardcoded file name
file_name = "/Users/user1/Desktop/JPL/PrometheusJPL/setupFiles/wasp39b-sodium.txt"

# Get the file information
file_info = get_file_information(file_name)

# Check if the file information is valid
if file_info:
    # Store the file information in variables
    file_path, planet_name, n_particles = file_info
    # Now you can use these variables as needed
else:
    print("Invalid file path!")


#print(get_file_information(file_name))
