import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from infoNa import get_file_information

# File path
file_path = '/Users/raghavchari/Desktop/output/Na.txt'

# Create the new file path by replacing "output" with "setupFiles"
new_file_path = file_path.replace("output", "setupFiles")

# Get the file information from info.py using the new file path
file_info = get_file_information(new_file_path)

# Read the file and store the lines
with open(file_path, 'r') as file:
    lines = file.readlines()

# Remove the first four lines
modified_lines = lines[4:]

# Parse the data for plotting
x_data = []
# Initialize empty lists for the 20 y-axis columns
y_data = [[] for _ in range(1)]

for line in modified_lines:
    values = line.strip().split()
    # Multiply x-axis values by 100000000
    x_data.append(float(values[0]) * 100000000)

    for i in range(1):
        # Multiply y-axis values by 1
        y_data[i].append(float(values[i + 1]) * 1)

# Create the plot
for i in range(1):
    # Set the legend label as "Col 1," "Col 2," etc.
    plt.plot(x_data, y_data[i], label=f'Col {i+1}')

# Set plot properties
plt.title('EXOMOON SODIUM TRANSIT SPECTRA')
plt.xlabel('Wavelength [Angstrom]')
plt.ylabel('Transmission')
plt.minorticks_on()
plt.grid(True, which='both', linestyle=':', linewidth=0.7, c='black')
plt.yscale('linear')

# Get the y-axis limits
y_min, y_max = plt.ylim()

# Calculate the y-value for the text annotation
text_height = 0.03 * (y_max - y_min)  # Set the height of the text annotation
# Adjust the height by subtracting a larger value
y_text = y_max - 0.6 * (y_max - y_min) - text_height

# Retrieve the planet name and n_particles from the file_info
planet_name = file_info[1]
n_particles = file_info[2]

# Create the legend label with the retrieved information
legend_label = f'Type: Exomoon\nPlanet Name: {planet_name}\nNa particles: {n_particles}'

# Add text annotation to the top left corner with a box
plt.text(0.02, y_text-0.02, legend_label, fontsize=10, transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor=(0.9, 0.9, 0.9), edgecolor='gray', pad=3), color='black')

# Display the legend on the right-hand side
plt.legend(loc='upper right')

# Modify the figure properties
plt.gcf().set_facecolor('lightgray')

# Modify the font family
plt.rcParams['font.family'] = 'serif'

# Show the plot
plt.show()
