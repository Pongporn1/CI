from mlp import MLP


HIDDEN_LAYERS = [8, 4]
LEARNING_RATE = 0.05
MOMENTUM = 0.5
MAX_EPOCHS = 5000
WEIGHT_SEED = 42


def main():
    """ทดสอบว่า MLP เรียนรู้ข้อมูลตัวอย่างหนึ่งชุดได้หรือไม่"""

    sample_input = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    sample_target = [0.35]

    network = MLP(
        input_size=8,
        hidden_layers=HIDDEN_LAYERS,
        output_size=1,
        learning_rate=LEARNING_RATE,
        momentum=MOMENTUM,
        seed=WEIGHT_SEED
    )

    print("=" * 32)
    print("ทดสอบ Multi Layer Perceptron")
    print("=" * 32)
    print("โครงสร้าง:", network.get_architecture())
    print("Learning rate:", LEARNING_RATE)
    print("Momentum:", MOMENTUM)
    print("Seed:", WEIGHT_SEED)

    prediction_before = network.predict(sample_input)[0]
    print("\nก่อน Training")
    print("Prediction:", round(prediction_before, 6))
    print("Target:", sample_target[0])

    print("\nเริ่ม Training")
    for epoch in range(1, MAX_EPOCHS + 1):
        mse, outputs = network.train_one(sample_input, sample_target)

        if epoch == 1 or epoch % 500 == 0:
            print(
                f"Epoch {epoch:5d} | "
                f"MSE = {mse:.10f} | "
                f"Prediction = {outputs[0]:.6f}"
            )

    prediction_after = network.predict(sample_input)[0]
    print("\nหลัง Training")
    print("Prediction:", round(prediction_after, 6))
    print("Target:", sample_target[0])
    print("ผลต่าง:", round(abs(sample_target[0] - prediction_after), 10))


if __name__ == "__main__":
    main()
