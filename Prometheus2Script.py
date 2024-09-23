import importlib
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from datetime import datetime
from scipy.special import wofz
import Prometheus.pythonScripts.gasProperties as gasprop
import Prometheus.pythonScripts.celestialBodies as bodies
import Prometheus.pythonScripts.geometryHandler as geom
import Prometheus.pythonScripts.constants as const

importlib.reload(gasprop)

def get_params(element, Nparticles):
    base_params = {
        "Fundamentals": {
            "ExomoonSource": True,
            "DopplerPlanetRotation": False,
            "CLV_variations": False,
            "RM_effect": False,
            "DopplerOrbitalMotion": True
        },
        "Architecture": {
            "planetName": "WASP-39b",
            "R_moon": 182200000.0,
            "a_moon": 15890367000.0,
            "starting_orbphase_moon": 1.5707963267948966
        },
        "Scenarios": {
            "exomoon": {
                "q_moon": 4.34
            }
        },
        "Species": {
            "exomoon": {
                element: {
                    "T": 1000,
                    "Nparticles": Nparticles
                }
            }
        },
        "Grids": {
            "lower_w": 1.5e-4,
            "upper_w": 9.5e-4,
            "resolutionLow": 0.01e-4,
            "widthHighRes": 0.1e-4,
            "resolutionHigh": 0.01e-4,
            "x_midpoint": 727056000000.0,
            "x_border": 532638000000.0,
            "x_steps": 50.0,
            "phi_steps": 50.0,
            "rho_steps": 50.0,
            "upper_rho": 5275680000,
            "orbphase_border": 0.,
            "orbphase_steps": 1.
        }
    }
    
    
    if element == "NaI":
        base_params["Species"]["exomoon"][element]["sigma_v"] = 2000000.0
        base_params["Grids"]["lower_w"] = 5.888e-05
        base_params["Grids"]["upper_w"] = 5.898e-05
        base_params["Grids"]["resolutionLow"] = 1e-08
        base_params["Grids"]["widthHighRes"] = 7.500000000000001e-09
        base_params["Grids"]["resolutionHigh"] = 1e-09
        base_params["Grids"]["x_steps"] = 50.0
        base_params["Grids"]["phi_steps"] = 50.0
        base_params["Grids"]["rho_steps"] = 50.0
        base_params["Grids"]["orbphase_border"] = 1.5707963267948966
        base_params["Grids"]["orbphase_steps"] = 50.0
    elif element == "SiO2":
        base_params["Species"]["exomoon"][element]["T"] = 1000.0
        base_params["Fundamentals"]["DopplerOrbitalMotion"] = True
        base_params["Grids"]["lower_w"] = 7e-4
        base_params["Grids"]["upper_w"] = 12e-4
    elif element == "KI":
        base_params["Species"]["exomoon"][element]["sigma_v"] = 1543288.733835272
        base_params["Architecture"]["a_moon"] = 17310735000.0
        base_params["Scenarios"]["exomoon"]["q_moon"] = 4.4
        base_params["Grids"]["lower_w"] = 7.665e-05
        base_params["Grids"]["upper_w"] = 7.703e-05
        base_params["Grids"]["resolutionLow"] = 1e-08
        base_params["Grids"]["widthHighRes"] = 7.500000000000001e-09
        base_params["Grids"]["resolutionHigh"] = 1e-09
        base_params["Grids"]["x_steps"] = 50.0
        base_params["Grids"]["phi_steps"] = 50.0
        base_params["Grids"]["rho_steps"] = 50.0
        base_params["Grids"]["orbphase_border"] = 1.5707963267948966
        base_params["Grids"]["orbphase_steps"] = 50.0
    return base_params

def voigt_profile(x, sigma, gamma):
    """ Voigt profile implementation """
    z = (x + 1j * gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

def extract_information(data, element):
    planet_name = data["Architecture"]["planetName"]
    n_particles = data["Species"]["exomoon"][element]["Nparticles"]
    return planet_name, n_particles

def plot_spectrum(file_path, element, planet_name, n_particles):
    # Read data from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    modified_lines = lines[4:]
    x_data = []
    y_data = [[] for _ in range(1)]
    
    for line in modified_lines:
        values = line.strip().split()
        x_data.append(float(values[0]) * 10000)  # Converting to microns here 
        for i in range(1):
            y_data[i].append((1 - float(values[i + 1])) * 1e6)  # Converting to ppm here
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    
    ax.plot(x_data, y_data[0], linewidth=2)
   
    title_text = f'EXOMOON {element} Transit Spectra - {planet_name}\n' \
                 f'{n_particles} {element} particles'
    ax.set_title(title_text, fontsize=16, fontweight='bold')
    
    ax.set_xlabel('Wavelength [Microns]', fontsize=14)
    ax.set_ylabel('Relative Transmission [ppm]', fontsize=14)
    
    ax.set_yscale('linear')
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.minorticks_on()
   
    fig.patch.set_facecolor('white')
    ax.set_facecolor('whitesmoke')
    
    legend_label = f'Type: Exomoon\nPlanet: {planet_name}\nParticles: {n_particles} {element}'
    ax.legend([legend_label], loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    
    plt.rcParams['font.family'] = 'serif'
    
    plt.show()

# Input from the user
element = input("Enter the molecule/atom (NaI, KI, SiO2, SO2): ")
Nparticles = float(input(f"Enter the number of particles for {element}: "))

params = get_params(element, Nparticles)

params_json = json.dumps(params, indent=4)
print(params_json)

fundamentalsDict = params['Fundamentals']
scenarioDict = params['Scenarios']
architectureDict = params['Architecture']
speciesDict = params['Species']
gridsDict = params['Grids']

planet = bodies.AvailablePlanets().findPlanet(architectureDict['planetName'])

wavelengthGrid = gasprop.WavelengthGrid(gridsDict['lower_w'], gridsDict['upper_w'], gridsDict['widthHighRes'], gridsDict['resolutionLow'], gridsDict['resolutionHigh'])
spatialGrid = geom.Grid(gridsDict['x_midpoint'], gridsDict['x_border'], int(gridsDict['x_steps']), gridsDict['upper_rho'], int(gridsDict['rho_steps']),
                        int(gridsDict['phi_steps']), gridsDict['orbphase_border'], int(gridsDict['orbphase_steps']))

scenarioList = []

for key_scenario in scenarioDict.keys():
    if key_scenario == 'barometric':
        scenarioList.append(gasprop.BarometricAtmosphere(scenarioDict['barometric']['T'], scenarioDict['barometric']['P_0'], scenarioDict['barometric']['mu'], planet))
    elif key_scenario == 'hydrostatic':
        scenarioList.append(gasprop.HydrostaticAtmosphere(scenarioDict['hydrostatic']['T'], scenarioDict['hydrostatic']['P_0'], scenarioDict['hydrostatic']['mu'], planet))
    elif key_scenario == 'powerLaw':
        if 'P_0' in scenarioDict['powerLaw'].keys():
            scenarioList.append(gasprop.PowerLawAtmosphere(scenarioDict['powerLaw']['T'], scenarioDict['powerLaw']['P_0'], scenarioDict['powerLaw']['q_esc'], planet))
        else:
            key_species = list(speciesDict['powerLaw'].keys())[0]  # Only one absorber
            Nparticles = speciesDict['powerLaw'][key_species]['Nparticles']
            scenarioList.append(gasprop.PowerLawExosphere(Nparticles, scenarioDict['powerLaw']['q_esc'], planet))
    elif key_scenario == 'exomoon':
        moon = bodies.Moon(architectureDict['starting_orbphase_moon'], architectureDict['R_moon'], architectureDict['a_moon'], planet)
        key_species = list(speciesDict['exomoon'].keys())[0]  # Only one absorber
        Nparticles = speciesDict['exomoon'][key_species]['Nparticles']
        scenarioList.append(gasprop.MoonExosphere(Nparticles, scenarioDict['exomoon']['q_moon'], moon))
    elif key_scenario == 'torus':
        key_species = list(speciesDict['torus'].keys())[0]  # Only one absorber
        Nparticles = speciesDict['torus'][key_species]['Nparticles']
        scenarioList.append(gasprop.TorusExosphere(Nparticles, scenarioDict['torus']['a_torus'], scenarioDict['torus']['v_ej'], planet))
    elif key_scenario == 'serpens':
        key_species = list(speciesDict['serpens'].keys())[0]  # Only one absorber
        Nparticles = speciesDict['serpens'][key_species]['Nparticles']
        scenario = gasprop.SerpensExosphere(scenarioDict['serpens']['serpensPath'], Nparticles, planet, 0.)
        scenario.addInterpolatedDensity(spatialGrid)
        scenarioList.append(scenario)  # sigmaSmoothing hardcoded to 0, i.e. no Gaussian smoothing of the serpens density distribution.

for idx, key_scenario in enumerate(scenarioDict.keys()):
    for key_species in speciesDict[key_scenario].keys():
        absorberDict = speciesDict[key_scenario][key_species]
        if key_species in const.AvailableSpecies().listSpeciesNames():  # Atom/ion
            if 'chi' in absorberDict:
                scenarioList[idx].addConstituent(key_species, absorberDict['chi'])
                scenarioList[idx].constituents[-1].addLookupFunctionToConstituent(wavelengthGrid)
            elif 'sigma_v' in absorberDict:
                scenarioList[idx].addConstituent(key_species, absorberDict['sigma_v'])
                scenarioList[idx].constituents[-1].addLookupFunctionToConstituent(wavelengthGrid)
        else:
            # Handle molecular species without opacity data
            scenarioList[idx].addMolecularConstituent(key_species, absorberDict['T'])
            scenarioList[idx].constituents[-1].addLookupFunctionToConstituent()

atmos = gasprop.Atmosphere(scenarioList, fundamentalsDict['DopplerOrbitalMotion'])

main = gasprop.Transit(atmos, wavelengthGrid, spatialGrid)
main.addWavelength()

R = main.sumOverChords()
wavelength = wavelengthGrid.constructWavelengthGrid(scenarioList)
orbphase = spatialGrid.constructOrbphaseAxis()

orbphaseOutput = np.insert(orbphase / (2. * np.pi), 0, np.nan)
mainOutput = np.vstack((wavelength, R))
output = np.vstack((orbphaseOutput, mainOutput.T))

header = 'Prometheus output file.\nFirst row: Orbital phases [1]\nAll other rows: Wavelength [cm] (first column), Transit depth R(orbital phase, wavelength) [1] (other columns)'

output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)
output_file_path = os.path.join(output_dir, f'{element}.txt')
np.savetxt(output_file_path, output, header=header)

elapsedTime = datetime.now() - datetime.now()

print("\nPROMETHEUS finished, yay! Elapsed time is:", elapsedTime)
print("The maximal flux decrease due to atmospheric/exospheric absorption in percent is:", np.abs(np.round(100 * (1 - np.min(R)), 10)))
print("The minimal flux decrease due to atmospheric/exospheric absorption in percent is:", np.abs(np.round(100 * (1 - np.max(R)), 10)))

planet_name, n_particles = extract_information(params, element)

plot_spectrum(output_file_path, element, planet_name, n_particles)
