from mlp import MLP, SimpleRandom


# ค่าที่ใช้ในการทดลอง
FILE_NAME = "flood.txt"
HIDDEN_LAYERS = [8, 4]
LEARNING_RATE = 0.10
MOMENTUM = 0.9
MAX_EPOCHS = 500
WEIGHT_SEED = 42
FOLD_SEED = 100
NUMBER_OF_FOLDS = 10


def load_data(file_name):
    """อ่านเฉพาะบรรทัดที่มีตัวเลข 9 ค่า"""
    data = []

    with open(file_name, "r", encoding="utf-8") as file:
        for line in file:
            values = line.split()

            if len(values) != 9:
                continue

            try:
                data.append([float(value) for value in values])
            except ValueError:
                pass

    return data


def split_xy(data):
    """8 ค่าแรกเป็น Input และค่าสุดท้ายเป็น Target"""
    inputs = [row[:8] for row in data]
    targets = [[row[8]] for row in data]
    return inputs, targets


def make_folds(data, number_of_folds, seed):
    """สุ่มข้อมูลแล้วแบ่งเป็น Fold"""
    copied_data = [row.copy() for row in data]
    rng = SimpleRandom(seed)
    rng.shuffle(copied_data)

    folds = []
    for fold_index in range(number_of_folds):
        folds.append(copied_data[fold_index::number_of_folds])

    return folds


def find_min_max(data):
    """หาค่าต่ำสุดและสูงสุดของแต่ละคอลัมน์"""
    minimums = data[0].copy()
    maximums = data[0].copy()

    for row in data:
        for i in range(len(row)):
            minimums[i] = min(minimums[i], row[i])
            maximums[i] = max(maximums[i], row[i])

    return minimums, maximums


def normalize(data, minimums, maximums):
    """แปลงข้อมูลให้อยู่ในช่วง 0 ถึง 1"""
    normalized_data = []

    for row in data:
        normalized_row = []

        for i in range(len(row)):
            value_range = maximums[i] - minimums[i]

            if value_range == 0:
                normalized_row.append(0.0)
            else:
                normalized_row.append(
                    (row[i] - minimums[i]) / value_range
                )

        normalized_data.append(normalized_row)

    return normalized_data


def denormalize(value, minimum, maximum):
    return value * (maximum - minimum) + minimum


def calculate_metrics(actual, predicted):
    squared_error = 0.0
    absolute_error = 0.0

    for actual_value, predicted_value in zip(actual, predicted):
        difference = actual_value - predicted_value
        squared_error += difference ** 2
        absolute_error += abs(difference)

    mse = squared_error / len(actual)
    rmse = mse ** 0.5
    mae = absolute_error / len(actual)
    return mse, rmse, mae


def average(values):
    return sum(values) / len(values)


def main():
    data = load_data(FILE_NAME)
    folds = make_folds(data, NUMBER_OF_FOLDS, FOLD_SEED)

    all_mse = []
    all_rmse = []
    all_mae = []

    print("Flood 10-Fold Cross-Validation")
    print("จำนวนข้อมูล:", len(data))
    print("Hidden layers:", HIDDEN_LAYERS)
    print("Learning rate:", LEARNING_RATE)
    print("Momentum:", MOMENTUM)
    print("Epochs:", MAX_EPOCHS)

    for test_fold in range(NUMBER_OF_FOLDS):
        test_data = folds[test_fold]
        train_data = []

        # ใช้ 1 Fold เป็น Test และอีก 9 Fold เป็น Train
        for fold_index in range(NUMBER_OF_FOLDS):
            if fold_index != test_fold:
                train_data.extend(folds[fold_index])

        train_inputs, train_targets = split_xy(train_data)
        test_inputs, test_targets = split_xy(test_data)

        # ต้องหา Min/Max จาก Training data เท่านั้น
        input_mins, input_maxs = find_min_max(train_inputs)
        target_mins, target_maxs = find_min_max(train_targets)

        train_inputs = normalize(train_inputs, input_mins, input_maxs)
        train_targets = normalize(train_targets, target_mins, target_maxs)
        test_inputs = normalize(test_inputs, input_mins, input_maxs)

        # สร้าง Network ใหม่ทุก Fold
        network = MLP(
            input_size=8,
            hidden_layers=HIDDEN_LAYERS,
            output_size=1,
            learning_rate=LEARNING_RATE,
            momentum=MOMENTUM,
            seed=WEIGHT_SEED
        )

        network.train(
            train_inputs,
            train_targets,
            max_epochs=MAX_EPOCHS,
            shuffle=True,
            seed=FOLD_SEED + test_fold,
            report_every=MAX_EPOCHS
        )

        actual_values = []
        predicted_values = []

        for inputs, target in zip(test_inputs, test_targets):
            normalized_prediction = network.predict(inputs)[0]
            prediction = denormalize(
                normalized_prediction,
                target_mins[0],
                target_maxs[0]
            )

            actual_values.append(target[0])
            predicted_values.append(prediction)

        mse, rmse, mae = calculate_metrics(actual_values, predicted_values)
        all_mse.append(mse)
        all_rmse.append(rmse)
        all_mae.append(mae)

        print(
            f"Fold {test_fold + 1:2d} | "
            f"RMSE = {rmse:.4f} | MAE = {mae:.4f}"
        )

    print("\nสรุปผล")
    print("Average MSE :", round(average(all_mse), 4))
    print("Average RMSE:", round(average(all_rmse), 4))
    print("Average MAE :", round(average(all_mae), 4))


if __name__ == "__main__":
    main()
