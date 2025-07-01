import os
import sys
import subprocess

streamlit_app_path = os.path.join("app", "app.py")

def main(): 
    # api_key = input('Input your Youtube API Key: ')

    command = [sys.executable, "-m", "streamlit", "run", streamlit_app_path] 

    process = subprocess.Popen(command)

if __name__ == '__main__':
    main()