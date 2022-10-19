import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default=15, dest='annealing', type=int)
    args = parser.parse_args()

    cities = pd.read_csv("Data/cities.csv")




if __name__ == '__main__':
    main()