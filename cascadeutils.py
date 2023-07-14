import os

def generate_negative_description_file():
    with open('neg.txt', 'w') as file:
        for filename in os.listdir(r'python_stuff\computer_vision\negative'):
            file.write(f'python_stuff\computer_vision\\negative\{filename}\n')

generate_negative_description_file()