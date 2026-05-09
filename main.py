from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train import train_model

def main():
    df = load_data()
    df = preprocess_data(df)
    train_model(df)

if __name__ == "__main__":
    main()