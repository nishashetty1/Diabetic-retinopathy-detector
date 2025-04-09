import os
import json

def create_kaggle_credentials():
    # Define the credentials structure
    credentials = {
        "username": input("Enter your Kaggle username: "),
        "key": input("Enter your Kaggle API key: ")
    }
    
    # Create the .kaggle directory in the correct location
    kaggle_dir = os.path.expanduser('~/.config/kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Write the credentials to kaggle.json
    kaggle_path = os.path.join(kaggle_dir, 'kaggle.json')
    with open(kaggle_path, 'w') as f:
        json.dump(credentials, f)
    
    # Set the correct permissions
    os.chmod(kaggle_path, 0o600)
    
    print(f"Kaggle credentials have been saved to {kaggle_path}")

if __name__ == "__main__":
    create_kaggle_credentials()