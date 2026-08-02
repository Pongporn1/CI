from mlp import MLP, SimpleRandom


# ค่าที่ใช้ในการทดลอง
FILE_NAME = "cross.pat"
HIDDEN_LAYERS = [8]
LEARNING_RATE = 0.1
MOMENTUM = 0.5
MAX_EPOCHS = 1000
WEIGHT_SEED = 42
FOLD_SEED = 100
NUMBER_OF_FOLDS = 10


def read_numbers(line):
    """ดึงเฉพาะตัวเลขออกจากหนึ่งบรรทัด"""
    numbers = []

    for part in line.replace(",", " ").split():
        try:
            numbers.append(float(part))
        except ValueError:
            pass

    return numbers


def load_data(file_name):
    """
    อ่านข้อมูล Cross

    รองรับทั้งรูปแบบ 4 ค่าในบรรทัดเดียว
    และรูปแบบ Input 2 ค่า กับ Target 2 ค่าแยกคนละบรรทัด
    """
    data = []
    pending_input = None

    with open(file_name, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            numbers = read_numbers(line)

            if len(numbers) >= 4:
                values = numbers[-4:]
                data.append([values[:2], values[2:]])
                pending_input = None

            elif len(numbers) == 2:
                if pending_input is None:
                    pending_input = numbers
                else:
                    data.append([pending_input, numbers])
                    pending_input = None

    return data


def class_number(values):
    """เลือก Class จาก Output ที่มีค่ามากกว่า"""
    if values[0] >= values[1]:
        return 0

    return 1


def make_folds(data, number_of_folds, seed):
    """แบ่งข้อมูลแบบรักษาสัดส่วนของแต่ละ Class"""
    class_0 = []
    class_1 = []

    for item in data:
        if class_number(item[1]) == 0:
            class_0.append(item)
        else:
            class_1.append(item)

    rng_0 = SimpleRandom(seed)
    rng_1 = SimpleRandom(seed + 1)
    rng_0.shuffle(class_0)
    rng_1.shuffle(class_1)

    folds = [[] for _ in range(number_of_folds)]

    for index, item in enumerate(class_0):
        folds[index % number_of_folds].append(item)

    for index, item in enumerate(class_1):
        folds[index % number_of_folds].append(item)

    return folds


def find_min_max(inputs):
    """หา Min และ Max ของ Input แต่ละตำแหน่ง"""
    minimums = inputs[0].copy()
    maximums = inputs[0].copy()

    for row in inputs:
        for i in range(len(row)):
            minimums[i] = min(minimums[i], row[i])
            maximums[i] = max(maximums[i], row[i])

    return minimums, maximums


def normalize(inputs, minimums, maximums):
    """แปลง Input ให้อยู่ในช่วง 0 ถึง 1"""
    result = []

    for row in inputs:
        normalized_row = []

        for i in range(len(row)):
            value_range = maximums[i] - minimums[i]

            if value_range == 0:
                normalized_row.append(0.0)
            else:
                normalized_row.append(
                    (row[i] - minimums[i]) / value_range
                )

        result.append(normalized_row)

    return result


def accuracy_from_matrix(matrix):
    correct = matrix[0][0] + matrix[1][1]
    total = sum(matrix[0]) + sum(matrix[1])

    if total == 0:
        return 0.0

    return correct / total * 100.0


def print_confusion_matrix(matrix):
    print("\nConfusion Matrix")
    print("              Predicted 0  Predicted 1")
    print(f"Actual 0      {matrix[0][0]:11d}  {matrix[0][1]:11d}")
    print(f"Actual 1      {matrix[1][0]:11d}  {matrix[1][1]:11d}")


def main():
    data = load_data(FILE_NAME)

    if len(data) == 0:
        print("ไม่พบข้อมูลใน cross.pat")
        print("ให้นำชุดข้อมูลจริงของอาจารย์มาใส่ในไฟล์ก่อน")
        return

    if len(data) < NUMBER_OF_FOLDS:
        print("จำนวนข้อมูลน้อยเกินไปสำหรับ 10-Fold Cross-Validation")
        return

    class_0_count = sum(1 for item in data if class_number(item[1]) == 0)
    class_1_count = len(data) - class_0_count
    folds = make_folds(data, NUMBER_OF_FOLDS, FOLD_SEED)

    total_matrix = [[0, 0], [0, 0]]
    fold_accuracies = []

    print("Cross 10-Fold Cross-Validation")
    print("จำนวนข้อมูล:", len(data))
    print("Class 0:", class_0_count, "| Class 1:", class_1_count)
    print("Hidden layers:", HIDDEN_LAYERS)
    print("Learning rate:", LEARNING_RATE)
    print("Momentum:", MOMENTUM)
    print("Epochs:", MAX_EPOCHS)
    print("Weight seed:", WEIGHT_SEED)

    for test_fold in range(NUMBER_OF_FOLDS):
        test_data = folds[test_fold]
        train_data = []

        for fold_index in range(NUMBER_OF_FOLDS):
            if fold_index != test_fold:
                train_data.extend(folds[fold_index])

        train_inputs = [item[0] for item in train_data]
        train_targets = [item[1] for item in train_data]
        test_inputs = [item[0] for item in test_data]
        test_targets = [item[1] for item in test_data]

        # หา Min/Max จาก Training data เท่านั้น
        input_mins, input_maxs = find_min_max(train_inputs)
        train_inputs = normalize(train_inputs, input_mins, input_maxs)
        test_inputs = normalize(test_inputs, input_mins, input_maxs)

        network = MLP(
            input_size=2,
            hidden_layers=HIDDEN_LAYERS,
            output_size=2,
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

        fold_matrix = [[0, 0], [0, 0]]

        for inputs, target in zip(test_inputs, test_targets):
            outputs = network.predict(inputs)
            actual_class = class_number(target)
            predicted_class = class_number(outputs)

            fold_matrix[actual_class][predicted_class] += 1
            total_matrix[actual_class][predicted_class] += 1

        fold_accuracy = accuracy_from_matrix(fold_matrix)
        fold_accuracies.append(fold_accuracy)

        print(
            f"Fold {test_fold + 1:2d} | "
            f"Accuracy = {fold_accuracy:.2f}%"
        )

    average_accuracy = sum(fold_accuracies) / len(fold_accuracies)

    print("\nสรุปผล")
    print("Average Accuracy:", round(average_accuracy, 2), "%")
    print("Overall Accuracy:", round(accuracy_from_matrix(total_matrix), 2), "%")
    print_confusion_matrix(total_matrix)


if __name__ == "__main__":
    main()
