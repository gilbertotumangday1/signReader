import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle

#make model
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

#normalize landmarks (to ensure it always works)
def normalize_landmarks(X):
    X_normalized = []
    for sample in X:
        normalized = []
        #locations of wrist coordinates from mediapipe hand
        wrist_x = sample[0]
        wrist_y = sample[21]
        wrist_z = sample[42]
        #normalize every aspect of the hand to make it so the wrist location (location of hand) is irrellevant
        for i in range(21):
            x = sample[i]     - wrist_x
            y = sample[i+21]  - wrist_y
            z = sample[i+42]  - wrist_z
            normalized.extend([x, y, z])
        X_normalized.append(normalized)
    #return numpy array of new normalized input
    return np.array(X_normalized, dtype=np.float32)

#if statement to ensure that we only train if the file itself has been run (rather than the MLP class being used)
if __name__ == "__main__":
    #load the dataset into a pandas dataframe
    df = pd.read_csv("train.csv")
    #get the raw independent data (drop the labels)
    X_raw = df.drop("label", axis=1).values.astype(np.float32)
    #get the dependent data (labels)
    y_raw = df["label"].values

    #create a label encoder to convert the letters to numbers 
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    # use normalize function above to ignore wrist location for all data
    X = normalize_landmarks(X_raw)

    #split train, validate and test data
    #first split the testing data from the training data
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    #then split the validate data from the training data
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1111, stratify=y_train_full, random_state=42)

    #create the x and y loaders for each data set, shuffling the training data
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long)), batch_size=32)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long)), batch_size=32)

    #create the model using the MLP class, specify loss function and learning rate for backprop
    model = MLP(input_size=63, num_classes=len(encoder.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #training loop 
    best_val_acc = 0.0
    for epoch in range(80):
        # Training
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad() #delete old gradients
            outputs = model(x_batch) #get output by sending training input into model
            loss = criterion(outputs, y_batch) #compute the loss of the labels vs predicted output
            loss.backward() #do backprop to get gradients
            optimizer.step() #update weights using lr
            running_loss += loss.item() #add to overall loss

        # Validation
        model.eval()
        #comparing predictions vs labels
        val_preds, val_labels = [], []
        with torch.no_grad(): #no gradient since only forward pass
            for x_batch, y_batch in val_loader: #now using validation set
                outputs = model(x_batch)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.numpy())
                val_labels.extend(y_batch.numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1} | Training Loss: {running_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

        # Save best model during training
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved (Val Acc: {best_val_acc:.4f})")

    #Now that training is done we can test on the test set
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader: #now using testing set
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.numpy())
            test_labels.extend(y_batch.numpy())


    test_accuracy = accuracy_score(test_labels, test_preds)
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")

    #Saving the model and label encoder
    saveStatus = input("Save final model as production version? (y): ")
    if saveStatus.lower() == "y":
        torch.save(model.state_dict(), "asl_mlp_model.pth")
        print("Model saved to asl_mlp_model.pth")

        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)
        print("Label encoder saved to label_encoder.pkl")
