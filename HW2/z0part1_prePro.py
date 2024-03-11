import csv

# Path to the input CSV file
csv_file_path = 'Assignment_2_modified_ Dataset.csv'

sentences = []  # This will hold the sentences (everything before the last word)
sentiments = []  # This will hold the sentiments (positive, negative, neutral)

# Open the CSV file for reading
with open(csv_file_path, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    
    # Iterate through each row in the CSV
    for row in reader:
        if not row:  # Skip empty rows
            continue
        
        # Assuming the sentiment is always the last word in the row
        sentiment = row[-1].strip().lower()  # Normalize to lower case for consistency
        
        # Check if the sentiment is one of the expected values
        if sentiment in ['positive', 'negative', 'neutral']:
            sentences.append(' '.join(row[:-1]))  # Everything except the last word
            sentiments.append(sentiment)
        else:
            print(f"Unexpected sentiment value '{sentiment}' found. Skipping this row.")

# Now we save the sentences and sentiments into two separate CSV files
# Save sentences to sentences.csv
with open('sentences.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for sentence in sentences:
        writer.writerow([sentence])  # Write each sentence as a row

# Save sentiments to sentiments.csv
with open('sentiments.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for sentiment in sentiments:
        writer.writerow([sentiment])  # Write each sentiment as a row

print("Files saved: sentences.csv, sentiments.csv")
