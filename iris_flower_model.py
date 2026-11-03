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
    # *Iris* flower classification with a neural network

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
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    return DataLoader, Dataset, F, pd, torch


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

    targetTensor = dfTensor[:, 4].to(torch.int64)
    dataTensor = dfTensor[:, :4].to(torch.float32)
    return dataTensor, targetTensor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Initialization

    Class definitions for the multilayer perceptron and the data batch loader.
    """)
    return


@app.cell
def _(Dataset, torch):
    class NeuralNetwork(torch.nn.Module):
        def __init__(self, num_inputs, num_outputs):
            super().__init__()

            layer_nodes = [40, 25, 10]

            # Define the layers with a Sequential
            self.layers = torch.nn.Sequential(

                # 1st hidden layer
                torch.nn.Linear(num_inputs, layer_nodes[0]),
                torch.nn.ReLU(),

                # 2nd hidden layer
                torch.nn.Linear(layer_nodes[0], layer_nodes[1]),
                torch.nn.ReLU(),

                # 3rd hidden layer
                torch.nn.Linear(layer_nodes[1], layer_nodes[2]),
                torch.nn.ReLU(),

                # Output layer
                torch.nn.Linear(layer_nodes[-1], num_outputs)
            )

        # Apply Sequential layers to inputs and return outputs
        def forward(self, x):
            logits = self.layers(x)
            return logits

    class IrisDataset(Dataset):
        def __init__(self, measurements, labels, device):
            self.measurementSets = measurements.to(device)
            self.labels = labels.to(device)

        def __getitem__(self, index):
            one_measurementSet = self.measurementSets[index]
            one_label = self.labels[index]
            return one_measurementSet, one_label

        def __len__(self):
            return self.labels.shape[0]

    return IrisDataset, NeuralNetwork


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Determine if any computational assets are avilable and initialize the IrisDataset, NeuralNetwork, and DataLoader on the fastest asset.
    """)
    return


@app.cell
def _(DataLoader, IrisDataset, dataTensor, targetTensor, torch):
    # Optimize to run on GPU resources if available
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    # Set the PyTorch seed for reproducible results
    # torch.manual_seed(123)

    iris_data_set = IrisDataset(dataTensor, targetTensor, device)

    iris_loader = DataLoader(
        dataset=iris_data_set,
        batch_size=30,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    return device, iris_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Loop
    The following hyperparameters were tuned based on experimental results:
    """)
    return


@app.cell
def _(NeuralNetwork, device):
    learning_rate = 0.001
    num_epochs = 2000

    model = NeuralNetwork(num_inputs=4, num_outputs=3).to(device)
    return learning_rate, model, num_epochs


@app.cell
def _(F, iris_loader, learning_rate, model, num_epochs, torch):
    # Stochastic gradient descent optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (measurementSets, labels) in enumerate(iris_loader):
            logits = model(measurementSets)

            # cross_entropy for stability as opposed to softmax
            loss = F.cross_entropy(logits, labels)

            # Prevent buildup of unused gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### LOGGING FOR EXPERIMENTAL TUNING
            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batch: {batch_idx+1:03d}/{len(iris_loader):03d}"
                  f" | Train Loss: {loss:2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy Testing
    """)
    return


@app.cell
def _(torch):
    def compute_accuracy(model, dataloader):
        model = model.eval()
        correct = 0
        total_examples = 0

        for idx, (measurementSets, labels) in enumerate(dataloader):

            with torch.no_grad():
                logits = model(measurementSets)

            predictions = torch.argmax(logits, dim=1)
            compare = labels == predictions
            correct += torch.sum(compare)
            total_examples += len(compare)

        return (correct/total_examples).item()

    return (compute_accuracy,)


@app.cell
def _(compute_accuracy, iris_loader, model):
    print(compute_accuracy(model, iris_loader))
    return


if __name__ == "__main__":
    app.run()
