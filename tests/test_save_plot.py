
import sys
import os
import pytest


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.save_plot import save_plot
import altair as alt
import pandas as pd
import seaborn as sns


from unittest.mock import MagicMock, patch  # for mocking file operations


import matplotlib.pyplot as plt

def test_save_plot():
    """Tests the save_plot function with an Altair plot and a Matplotlib plot."""
    directory = str(os.getcwd() + "/../test/")  # Set the directory to the current working directory

    # Read the data
    df = pd.read_csv("DATA/laptops.csv")
    
    #first graph: brand distribution
    categorical = df.select_dtypes(include=['object'])
    plt.figure(figsize=(13,7))
    brand_counts = categorical.brand.value_counts()
    axis = sns.barplot(x=brand_counts.index, y=brand_counts.values)
    axis.bar_label(axis.containers[0], fontsize=7)
    plt.xlabel('Brand')
    plt.ylabel('Count')
    plt.title('Brand distribution')
    plt.xticks(rotation=45)

    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    plt_obj = plt.gcf()
    save_plot(plt_obj, 'test_plt.png', dir)

    # ALTAIR

    #second graph: price distributio plot
    df['Price'] = df['Price'].astype(int)

    # Create a histogram of laptop prices
    chart = alt.Chart(df).mark_bar(
        color='#7ca0ff',  # Bar color
        opacity=0.7       # Bar opacity
    ).encode(
        x=alt.X('Price:Q', bin=alt.Bin(maxbins=40), title='Price (USD)'),
        y=alt.Y('count()', title='Frequency'),
        tooltip=[alt.Tooltip('count()', title='Frequency'), alt.Tooltip('Price:Q', title='Price Range')]
    ).properties(
        title='Distribution of Laptop Prices',
        width=600,
        height=400
    )

    save_plot(chart, 'test_altair.png', dir)



def test_save_plot_empty_filename():
  """Tests the save_plot function with an empty filename."""
  tmpdir = ""
  mock_plot = MagicMock()
  directory = tmpdir

  with pytest.raises(ValueError):
    save_plot(mock_plot, "", directory)

def test_save_plot_nonexistent_directory():
  """Tests the save_plot function with a non-existent directory."""
  mock_plot = MagicMock()
  filename = "test_plot"
  directory = "nonexistent_dir"

  with pytest.raises(OSError):
    save_plot(mock_plot, filename, directory)

test_save_plot()
test_save_plot_empty_filename()
test_save_plot_nonexistent_directory()

