# ค่า e สำหรับใช้ใน Sigmoid
E = 2.718281828459045


class SimpleRandom:
    """ตัวสร้างเลขสุ่มแบบง่าย ใช้ seed เดิมจะได้ผลเดิม"""

    def __init__(self, seed=1):
        self.state = int(seed) % 4294967296

    def next_number(self):
        self.state = (1664525 * self.state + 1013904223) % 4294967296
        return self.state / 4294967296

    def uniform(self, minimum, maximum):
        return minimum + (maximum - minimum) * self.next_number()

    def shuffle(self, values):
        # Fisher-Yates shuffle
        for i in range(len(values) - 1, 0, -1):
            j = int(self.next_number() * (i + 1))
            values[i], values[j] = values[j], values[i]


class MLP:
    """Multi-Layer Perceptron ที่รองรับ Hidden layer หลายชั้น"""

    def __init__(
        self,
        input_size,
        output_size,
        learning_rate=0.01,
        momentum=0.0,
        seed=1,
        hidden_layers=None,
        hidden_size=None
    ):
        # hidden_size เก็บไว้เพื่อรองรับโค้ดแบบเดิม
        if hidden_layers is None:
            if hidden_size is None:
                raise ValueError("ต้องกำหนด hidden_layers หรือ hidden_size")
            hidden_layers = [hidden_size]

        if input_size <= 0 or output_size <= 0:
            raise ValueError("จำนวน Input และ Output ต้องมากกว่า 0")

        if not hidden_layers or any(size <= 0 for size in hidden_layers):
            raise ValueError("จำนวน Hidden nodes ต้องมากกว่า 0")

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers.copy()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.layer_sizes = [input_size] + self.hidden_layers + [output_size]

        rng = SimpleRandom(seed)
        self.weights = []
        self.biases = []
        self.previous_weight_changes = []
        self.previous_bias_changes = []

        # สร้าง Weight และ Bias ของทุกคู่ Layer
        for layer in range(len(self.layer_sizes) - 1):
            from_size = self.layer_sizes[layer]
            to_size = self.layer_sizes[layer + 1]

            weight_matrix = []
            change_matrix = []

            for _ in range(from_size):
                weight_matrix.append([
                    rng.uniform(-0.5, 0.5) for _ in range(to_size)
                ])
                change_matrix.append([0.0 for _ in range(to_size)])

            self.weights.append(weight_matrix)
            self.biases.append([
                rng.uniform(-0.5, 0.5) for _ in range(to_size)
            ])
            self.previous_weight_changes.append(change_matrix)
            self.previous_bias_changes.append([0.0 for _ in range(to_size)])

        self.activations = None

    def sigmoid(self, value):
        if value <= -60:
            return 0.0
        if value >= 60:
            return 1.0
        return 1.0 / (1.0 + E ** (-value))

    def sigmoid_derivative(self, output):
        return output * (1.0 - output)

    def get_architecture(self):
        return " -> ".join(str(size) for size in self.layer_sizes)

    def forward(self, inputs):
        """คำนวณ Output จาก Input ไปข้างหน้า"""
        if len(inputs) != self.input_size:
            raise ValueError(
                f"ต้องการ Input {self.input_size} ค่า แต่ได้รับ {len(inputs)} ค่า"
            )

        self.activations = [inputs.copy()]

        for layer in range(len(self.weights)):
            previous_outputs = self.activations[-1]
            current_outputs = []

            for current_node in range(len(self.biases[layer])):
                total = self.biases[layer][current_node]

                for previous_node in range(len(previous_outputs)):
                    total += (
                        previous_outputs[previous_node]
                        * self.weights[layer][previous_node][current_node]
                    )

                current_outputs.append(self.sigmoid(total))

            self.activations.append(current_outputs)

        return self.activations[-1]

    def backward(self, targets):
        """คำนวณ Delta ย้อนกลับและปรับ Weight ด้วย Momentum"""
        if self.activations is None:
            raise ValueError("ต้องทำ forward() ก่อน backward()")

        if len(targets) != self.output_size:
            raise ValueError(
                f"ต้องการ Target {self.output_size} ค่า แต่ได้รับ {len(targets)} ค่า"
            )

        number_of_layers = len(self.weights)
        deltas = [None for _ in range(number_of_layers)]

        # Output delta
        outputs = self.activations[-1]
        deltas[-1] = []

        for i in range(self.output_size):
            error = targets[i] - outputs[i]
            deltas[-1].append(error * self.sigmoid_derivative(outputs[i]))

        # Hidden delta คำนวณย้อนกลับทีละชั้น
        for layer in range(number_of_layers - 2, -1, -1):
            current_outputs = self.activations[layer + 1]
            next_deltas = deltas[layer + 1]
            next_weights = self.weights[layer + 1]
            current_deltas = []

            for current_node in range(len(current_outputs)):
                error = 0.0

                for next_node in range(len(next_deltas)):
                    error += (
                        next_deltas[next_node]
                        * next_weights[current_node][next_node]
                    )

                current_deltas.append(
                    error * self.sigmoid_derivative(current_outputs[current_node])
                )

            deltas[layer] = current_deltas

        # ปรับ Weight และ Bias ทุกชั้น
        for layer in range(number_of_layers):
            previous_outputs = self.activations[layer]

            for previous_node in range(len(previous_outputs)):
                for current_node in range(len(deltas[layer])):
                    previous_change = self.previous_weight_changes[
                        layer
                    ][previous_node][current_node]

                    change = (
                        self.learning_rate
                        * deltas[layer][current_node]
                        * previous_outputs[previous_node]
                        + self.momentum * previous_change
                    )

                    self.weights[layer][previous_node][current_node] += change
                    self.previous_weight_changes[
                        layer
                    ][previous_node][current_node] = change

            for current_node in range(len(deltas[layer])):
                previous_change = self.previous_bias_changes[layer][current_node]
                change = (
                    self.learning_rate * deltas[layer][current_node]
                    + self.momentum * previous_change
                )

                self.biases[layer][current_node] += change
                self.previous_bias_changes[layer][current_node] = change

    def train_one(self, inputs, targets):
        """ฝึก Network ด้วยข้อมูลหนึ่งตัวอย่าง"""
        outputs = self.forward(inputs)

        squared_error = 0.0
        for i in range(self.output_size):
            squared_error += (targets[i] - outputs[i]) ** 2

        mse = squared_error / self.output_size
        self.backward(targets)
        return mse, outputs

    def predict(self, inputs):
        return self.forward(inputs)

    def train(
        self,
        training_inputs,
        training_targets,
        max_epochs=2000,
        shuffle=True,
        seed=1,
        report_every=100,
        error_threshold=None
    ):
        """ฝึก Network ด้วย Training data ทั้งหมด"""
        if len(training_inputs) != len(training_targets):
            raise ValueError("จำนวน Input และ Target ไม่เท่ากัน")
        if not training_inputs:
            raise ValueError("ไม่มี Training data")

        rng = SimpleRandom(seed)
        indices = list(range(len(training_inputs)))
        error_history = []

        for epoch in range(1, max_epochs + 1):
            if shuffle:
                rng.shuffle(indices)

            total_mse = 0.0
            for data_index in indices:
                mse, _ = self.train_one(
                    training_inputs[data_index],
                    training_targets[data_index]
                )
                total_mse += mse

            average_mse = total_mse / len(training_inputs)
            error_history.append(average_mse)

            if (
                epoch == 1
                or epoch == max_epochs
                or (report_every > 0 and epoch % report_every == 0)
            ):
                print(
                    f"Epoch {epoch:5d} | Training MSE = {average_mse:.8f}"
                )

            if (
                error_threshold is not None
                and average_mse <= error_threshold
            ):
                print("หยุดก่อนครบ Epoch เพราะ MSE ต่ำกว่า Threshold")
                break

        return error_history
