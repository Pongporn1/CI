from mlp import MLP, SimpleRandom


# ค่าที่ใช้ในการทดลอง สามารถเปลี่ยนได้จากตรงนี้
FILE_NAME = "flood.txt"
HIDDEN_LAYERS = [8, 4]
LEARNING_RATE = 0.10
MOMENTUM = 0.9
MAX_EPOCHS = 500
WEIGHT_SEED = 42
FOLD_SEED = 100
NUMBER_OF_FOLDS = 10


def load_data(file_name):
    """อ่าน Flood data โดยเก็บเฉพาะบรรทัดที่มีตัวเลข 9 ค่า"""
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
    """สุ่มข้อมูลแล้วแบ่งเป็น Folds"""
    copied_data = [row.copy() for row in data]

    random_generator = SimpleRandom(seed)
    random_generator.shuffle(copied_data)

    folds = []

    for fold_index in range(number_of_folds):
        folds.append(
            copied_data[fold_index::number_of_folds]
        )

    return folds


def find_min_max(data):
    """หาค่าต่ำสุดและสูงสุดของแต่ละ Feature"""
    minimums = data[0].copy()
    maximums = data[0].copy()

    for row in data:
        for index in range(len(row)):
            if row[index] < minimums[index]:
                minimums[index] = row[index]

            if row[index] > maximums[index]:
                maximums[index] = row[index]

    return minimums, maximums


def normalize(data, minimums, maximums):
    """ทำ Min-Max normalization"""
    result = []

    for row in data:
        normalized_row = []

        for index in range(len(row)):
            value_range = maximums[index] - minimums[index]

            if value_range == 0:
                normalized_value = 0.0
            else:
                normalized_value = (
                    row[index] - minimums[index]
                ) / value_range

            normalized_row.append(normalized_value)

        result.append(normalized_row)

    return result


def denormalize(value, minimum, maximum):
    """แปลงค่าที่โมเดลทำนายกลับเป็นระดับน้ำจริง"""
    return value * (maximum - minimum) + minimum


def calculate_metrics(actual, predicted):
    """คำนวณ MSE, RMSE และ MAE"""
    squared_error = 0.0
    absolute_error = 0.0

    for index in range(len(actual)):
        difference = actual[index] - predicted[index]

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
    folds = make_folds(
        data,
        NUMBER_OF_FOLDS,
        FOLD_SEED
    )

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
        # Fold ปัจจุบันเป็น Test
        test_data = folds[test_fold]

        # Fold ที่เหลือเป็น Train
        train_data = []

        for fold_index in range(NUMBER_OF_FOLDS):
            if fold_index != test_fold:
                train_data.extend(folds[fold_index])

        train_inputs, train_targets = split_xy(train_data)
        test_inputs, test_targets = split_xy(test_data)

        # หา Min/Max จาก Training data เท่านั้น
        input_mins, input_maxs = find_min_max(train_inputs)
        target_mins, target_maxs = find_min_max(train_targets)

        normalized_train_inputs = normalize(
            train_inputs,
            input_mins,
            input_maxs
        )

        normalized_train_targets = normalize(
            train_targets,
            target_mins,
            target_maxs
        )

        normalized_test_inputs = normalize(
            test_inputs,
            input_mins,
            input_maxs
        )

        # แต่ละ Fold ต้องสร้าง Network ใหม่
        network = MLP(
            input_size=8,
            hidden_layers=HIDDEN_LAYERS,
            output_size=1,
            learning_rate=LEARNING_RATE,
            momentum=MOMENTUM,
            seed=WEIGHT_SEED
        )

        network.train(
            normalized_train_inputs,
            normalized_train_targets,
            max_epochs=MAX_EPOCHS,
            shuffle=True,
            seed=FOLD_SEED + test_fold,
            report_every=MAX_EPOCHS
        )

        actual_values = []
        predicted_values = []

        for index in range(len(normalized_test_inputs)):
            normalized_prediction = network.predict(
                normalized_test_inputs[index]
            )[0]

            prediction = denormalize(
                normalized_prediction,
                target_mins[0],
                target_maxs[0]
            )

            actual_values.append(test_targets[index][0])
            predicted_values.append(prediction)

        mse, rmse, mae = calculate_metrics(
            actual_values,
            predicted_values
        )

        all_mse.append(mse)
        all_rmse.append(rmse)
        all_mae.append(mae)

        print(
            f"Fold {test_fold + 1:2d} | "
            f"RMSE = {rmse:.4f} | "
            f"MAE = {mae:.4f}"
        )

    print("\nสรุปผล")
    print("Average MSE :", round(average(all_mse), 4))
    print("Average RMSE:", round(average(all_rmse), 4))
    print("Average MAE :", round(average(all_mae), 4))


if __name__ == "__main__":
    main()