import os


def save_plot(plot, filename, directory):
  """Saves a plot object to a file in the specified directory.

  Args:
      plot: The plot object to save (e.g., matplotlib plot or Altair chart).
      filename: The filename for the saved plot (without extension).
      directory: The directory path to save the plot in.
  """

  # Save plot with absolute path
  output_path = os.path.join(directory, filename)
  if (filename == ""):
    raise ValueError("No filename provided.")
  
  if not os.path.exists(directory):
    raise OSError(f"Directory '{directory}' does not exist.")
  

  try:
    plot.savefig(output_path)
  except:
    plot.save(output_path, scale_factor=2.0)

