from ultralytics import YOLO


def main():
    model = YOLO("best.pt")

    results = model.predict(
        source="amostra.jpg",
        save=True,
        conf=0.25
    )

    print("Inferência concluída.")


if __name__ == "__main__":
    main()