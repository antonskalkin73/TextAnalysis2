import csv
import os
import pandas as pd

def main():
    with open('Data/LabeledTestData.csv', "w", newline="", encoding="utf8") as file:
        columns = ["id", "sentiment", "review"]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        directPos = os.listdir('TXT/Positiv')
        directNeg = os.listdir('TXT/Negativ')
        id = 0
        for name in directPos:
            id += 1
            with open('TXT/Positiv/' + name, "r") as file:
                content = file.read().replace('\n', '')
                writer.writerow({"id" : id, "sentiment" : 1, "review" : content})
        id = 1000
        for name in directNeg:
            id += 1
            with open('TXT/Negativ/' + name, "r") as file:
                content = file.read().replace('\n', '')
                writer.writerow({"id" : id, "sentiment" : 0, "review" : content})

    df = pd.read_csv('Data/LabeledTestData.csv')
    df = df.sample(frac=1)
    df.to_csv('Data/LabeledTestData.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    main()