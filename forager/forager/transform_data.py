import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def transform_data(csv_file_path):
    """
    Transforms participant response data to the format needed to run forager 

    Original:
    9,1,Apple ,ant ,aardvark,amazing ,avast

    Transformed:
    9,Apple
    9,ant
    9,aardvark
    9,amazing
    9,avast
    """
    try: 
        #extracting file name from the path
        csv_name = os.path.basename(csv_file_path).split('.')[0]

        #load csv line by line
        with open(csv_file_path, 'r') as file:
            raw_data = file.readlines()

        data = []
        for line in raw_data[1:]:  #skipping the header 
            line = line.strip()  
            items = line.split(',') 
            participant = items[0]
            for response in items[2:]:
                cleaned_response = response.strip().strip('"')  # remove both whitespace and double quotes
                if cleaned_response:  # ensure response is not empty
                    data.append([participant, cleaned_response])

        #turning into df
        df = pd.DataFrame(data, columns=['Participant', 'Response'])

        if df.empty:
            print("No valid data found in the CSV file.")
            return

        unique_participants = df['Participant'].unique()
        num_participants = len(unique_participants)
        print(f"{csv_name}'s number of participants: {num_participants}")

        # New csv
        output_folder = "forager/data/fluency_lists/transformed_data"
        df.to_csv(os.path.join(output_folder, f'{csv_name}_transformed.csv'), index=False)

    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except Exception as e:
        print(f"Unknown error: {e}")
    


