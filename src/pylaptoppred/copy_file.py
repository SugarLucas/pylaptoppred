import click
import os


def copy_file(input_path, output_path):
    """This function copy a file from the input path to the output path.

    Args:
        input_path (str): Path of the file to be read.
        output_path (str): Path where the file will be saved.
    """
    try:
        with open(input_path, 'r') as file:
            data = file.read()
        
        with open(output_path, 'w') as file:
            file.write(data)

        click.echo(f"File successfully copied from {input_path} to {output_path}")
    except Exception as e:
        click.echo(f"Failed to copy the file: {e}")




if __name__ == '__main__':
    copy_file()
