from mlp import MLP


# ==========================================
# NETWORK TEST CONFIGURATION
# ==========================================

# [8] หมายถึง Hidden layer 1 ชั้น มี 8 nodes
# [8, 4] หมายถึง Hidden layer 2 ชั้น
HIDDEN_LAYERS = [8,4]

LEARNING_RATE = 0.05
MOMENTUM = 0.5
MAX_EPOCHS = 5000
WEIGHT_SEED = 42


def main():
    """
    ทดสอบว่า MLP หลาย Hidden layers
    สามารถเรียนรู้ข้อมูลหนึ่งตัวอย่างได้หรือไม่
    """

    # Input จำลองจำนวน 8 features
    sample_input = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80
    ]

    # Target จำลองที่ Normalize แล้ว
    sample_target = [0.35]

    network = MLP(
        input_size=8,
        hidden_layers=HIDDEN_LAYERS,
        output_size=1,
        learning_rate=LEARNING_RATE,
        momentum=MOMENTUM,
        seed=WEIGHT_SEED
    )

    print("================================")
    print("ทดสอบ Multi Layer Perceptron")
    print("================================")
    print(
        "โครงสร้าง:",
        network.get_architecture()
    )
    print("Learning rate:", LEARNING_RATE)
    print("Momentum:", MOMENTUM)
    print("Seed:", WEIGHT_SEED)

    # ทำนายก่อน Training
    prediction_before = network.predict(
        sample_input
    )[0]

    print()
    print("ก่อน Training")
    print(
        "Prediction:",
        round(prediction_before, 6)
    )
    print(
        "Target:",
        sample_target[0]
    )

    print()
    print("เริ่ม Training")

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):
        mse, outputs = network.train_one(
            sample_input,
            sample_target
        )

        if (
            epoch == 1
            or epoch % 500 == 0
        ):
            print(
                f"Epoch {epoch:5d} | "
                f"MSE = {mse:.10f} | "
                f"Prediction = {outputs[0]:.6f}"
            )

    # ทำนายหลัง Training
    prediction_after = network.predict(
        sample_input
    )[0]

    print()
    print("หลัง Training")
    print(
        "Prediction:",
        round(prediction_after, 6)
    )
    print(
        "Target:",
        sample_target[0]
    )
    print(
        "ผลต่าง:",
        round(
            abs(
                sample_target[0]
                - prediction_after
            ),
            10
        )
    )


if __name__ == "__main__":
    main()