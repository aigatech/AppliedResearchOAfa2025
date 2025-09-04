# Movie Recommendation System

## What it does
This project is an interactive movie recommendation system built with Python and JupyterLab. It allows users to type a movie title and get recommendations based on users with similar tastes. The recommendations are generated using TF-IDF text similarity for searching titles and collaborative filtering for suggesting movies.

## How to run it
1. Install required packages:
```bash
pip install datasets pandas numpy scikit-learn ipywidgets
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```
2. Open the Jupyter Notebook or JupyterLab environment.
3. Run the notebook cells in order.
4. Use the interactive text box to type a movie title (at least 5 characters).
5. Recommendations will appear below the input box.
