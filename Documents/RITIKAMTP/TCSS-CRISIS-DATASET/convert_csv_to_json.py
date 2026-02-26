import csv
import json
import os

input_csv = 'BangaloreRiots.csv'
output_json = 'tweetsample.json'

def convert_csv_to_json():
    # Only try to open if the CSV file exists
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found in the current directory.")
        return

    tweets = []
    
    try:
        with open(input_csv, mode='r', encoding='utf-8', errors='replace') as infile:
            reader = csv.DictReader(infile)
            
            # Check for 'Tweet' column
            if 'Tweet' not in reader.fieldnames:
                print(f"Error: 'Tweet' column not found in CSV. Found: {reader.fieldnames}")
                return
            
            for row in reader:
                tweet_text = row.get('Tweet', '').strip()
                if tweet_text:
                    tweets.append(tweet_text)
        
        # Write to JSON file
        data = {'tweets': tweets}
        with open(output_json, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False)
            
        print(f"Successfully converted {len(tweets)} tweets to {output_json}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    convert_csv_to_json()
