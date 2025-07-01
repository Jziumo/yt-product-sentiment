import os

output_dir = './output'

def output_to_file(filename, text, mode='w'):
    """
    Write the message to the specified file. 
    """
    with open(os.path.join(output_dir, filename), mode) as file:
        file.write(text)