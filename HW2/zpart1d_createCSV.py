import json
import csv

# Define the file name
input_file_name = '1.json'
output_file_name = 'output.csv'

# Open and read the JSON file
with open(input_file_name, 'r') as file:
    # Load JSON data
    data = json.load(file)

# Define field names for the CSV file
field_names = ["trainingAccuracy", "testAccuracy"]

# Write data to CSV file
with open(output_file_name, 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=field_names)
    writer.writeheader()

    # Assuming data is an array of objects, you can iterate over each object
    for i in range(5000):
        writer.writerow({"trainingAccuracy": data[i]["trainingAccuracy"], "testAccuracy": data[i]["testAccuracy"]})

# Define the file name
input_file_name = '5k.json'

# Open and read the JSON file
with open(input_file_name, 'r') as file:
    # Load JSON data
    data = json.load(file)

# Write data to CSV file
with open(output_file_name, 'a', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=field_names)

    try:
        for i in range(5000):
            writer.writerow({"trainingAccuracy": data[i+5000]["trainingAccuracy"], "testAccuracy": data[i+5000]["testAccuracy"]})
    except:
        pass
