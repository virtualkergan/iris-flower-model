import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # *Iris* flower classification with neural network

    This model is trained on the *Iris* flower data set used in Ronald Fisher's 1936 paper, *The use of multiple measurements in taxonomic problems*. The concepts used here were learned from Appendix A of Sebastian Raschka's book, *Build a Large Language Model (From Scratch)*.

    The data set has instances that belong to one of three classifications of *Iris* flowers:

    1. *I. setosa*
    2. *I. versicolor*
    3. *I. virginica*

    For each instance, four measurements are recorded:

    1. Sepal length
    2. Sepal width
    3. Petal length
    4. Petal width

    This model is trained to classify an *Iris* flower based on its four provided measurements.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Module Imports
    """)
    return


@app.cell
def _():
    import pandas as pd
    import torch

    return pd, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Preprocessing

    Separating the data into two PyTorch tensors makes training the model simpler. In addition, mapping the names of the *Iris* flowers to numerical labels will match the indexing method used by PyTorch and aid in writing the training method.
    """)
    return


@app.cell
def _(pd, torch):
    # Constant definitions
    FILE_NAME = 'data/iris.data'
    DATA_LABELS = ['sepal_length',
                   'sepal_width',
                   'petal_length',
                   'petal_width',
                   'class']
    TARGET_LABELS = ['Iris-setosa',
                    'Iris-versicolor',
                    'Iris-virginica']

    # Read in raw data from CSV file
    df = pd.read_csv('data/iris.data', header=None)

    # Assign column names to pandas dataframe
    df = df.rename(columns={0: DATA_LABELS[0],
                            1: DATA_LABELS[1],
                            2: DATA_LABELS[2],
                            3: DATA_LABELS[3],
                            4: DATA_LABELS[4]})

    # Map target labels to indices
    df['label'] = 2
    for i in range(0, df.shape[0]):
        if df.at[i, 'class'] == TARGET_LABELS[0]:
            df.at[i, 'label'] = 0
        elif df.at[i, 'class'] == TARGET_LABELS[1]:
            df.at[i, 'label'] = 1
        else:
            continue

    # Remove string labels for PyTorch conversion
    df = df.drop('class', axis=1)

    # Convert to PyTorch tensors
    dfTensor = torch.from_numpy(df.values)

    targetTensor = dfTensor[:, 4]
    dataTensor = dfTensor[:, :4]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Initialization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Loop
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy Testing
    """)
    return


if __name__ == "__main__":
    app.run()
