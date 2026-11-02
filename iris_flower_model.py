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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Preprocessing
    """)
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
