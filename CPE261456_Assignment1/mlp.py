# ค่าประมาณของเลข e สำหรับใช้ใน Sigmoid
# เขียนไว้เองเพื่อไม่ต้อง import math
E = 2.718281828459045


class SimpleRandom:
    """
    เครื่องสร้างเลขสุ่มอย่างง่าย

    Seed เดิมจะได้ลำดับเลขสุ่มเดิม
    Seed ต่างกันจะทำให้ Weight เริ่มต้นต่างกัน
    """

    def __init__(self, seed):
        # เก็บสถานะปัจจุบันของตัวสร้างเลขสุ่ม
        self.state = int(seed) % 4294967296

    def next_number(self):
        """
        สร้างเลขทศนิยมตั้งแต่ 0 ถึงน้อยกว่า 1
        ด้วยวิธี Linear Congruential Generator
        """

        self.state = (
            1664525 * self.state + 1013904223
        ) % 4294967296

        return self.state / 4294967296

    def uniform(self, minimum, maximum):
        """
        สร้างเลขสุ่มในช่วง minimum ถึง maximum
        """

        return minimum + (
            maximum - minimum
        ) * self.next_number()

    def shuffle(self, values):
        """
        สลับตำแหน่งข้อมูลด้วยวิธี Fisher-Yates
        """

        for index in range(
            len(values) - 1,
            0,
            -1
        ):
            swap_index = int(
                self.next_number() * (index + 1)
            )

            values[index], values[swap_index] = (
                values[swap_index],
                values[index]
            )


class MLP:
    """
    Multi Layer Perceptron ที่รองรับ Hidden layer หลายชั้น

    ตัวอย่าง:

    hidden_layers=[8]
    หมายถึง Input -> Hidden 8 nodes -> Output

    hidden_layers=[8, 4]
    หมายถึง Input -> Hidden 8 nodes
    -> Hidden 4 nodes -> Output
    """

    def __init__(
        self,
        input_size,
        hidden_size=None,
        output_size=None,
        learning_rate=0.01,
        momentum=0.0,
        seed=1,
        hidden_layers=None
    ):
        """
        สร้างโครงสร้าง Network พร้อมสุ่ม Weight และ Bias

        รองรับโค้ดเดิมที่ใช้ hidden_size=8
        และโค้ดใหม่ที่ใช้ hidden_layers=[8, 4]
        """

        if input_size <= 0:
            raise ValueError(
                "input_size ต้องมากกว่า 0"
            )

        if output_size is None or output_size <= 0:
            raise ValueError(
                "output_size ต้องมากกว่า 0"
            )

        # ถ้าไม่ได้ส่ง hidden_layers มา
        # ให้เปลี่ยน hidden_size เดิมเป็น List
        if hidden_layers is None:
            if hidden_size is None:
                raise ValueError(
                    "ต้องกำหนด hidden_size "
                    "หรือ hidden_layers"
                )

            hidden_layers = [hidden_size]

        if len(hidden_layers) == 0:
            raise ValueError(
                "MLP ต้องมี Hidden layer "
                "อย่างน้อย 1 ชั้น"
            )

        for node_count in hidden_layers:
            if node_count <= 0:
                raise ValueError(
                    "จำนวน Hidden nodes "
                    "ต้องมากกว่า 0"
                )

        self.input_size = input_size
        self.hidden_layers = hidden_layers.copy()
        self.output_size = output_size

        self.learning_rate = learning_rate
        self.momentum = momentum

        # ตัวอย่าง:
        # input=8, hidden=[8, 4], output=1
        # จะได้ [8, 8, 4, 1]
        self.layer_sizes = (
            [input_size]
            + self.hidden_layers
            + [output_size]
        )

        # ตัวสร้างเลขสุ่มสำหรับ Weight และ Bias
        self.random_generator = SimpleRandom(seed)

        # weights[layer][node_from][node_to]
        self.weights = []

        # biases[layer][node_to]
        self.biases = []

        # เก็บค่าเปลี่ยนแปลงรอบก่อน
        # เพื่อใช้คำนวณ Momentum
        self.previous_weight_changes = []
        self.previous_bias_changes = []

        number_of_weight_layers = (
            len(self.layer_sizes) - 1
        )

        # สร้าง Weight และ Bias ของทุกคู่ Layer
        for layer_index in range(
            number_of_weight_layers
        ):
            from_size = self.layer_sizes[
                layer_index
            ]

            to_size = self.layer_sizes[
                layer_index + 1
            ]

            weight_matrix = []
            previous_change_matrix = []

            # สร้าง Weight Matrix
            for from_node in range(from_size):
                weight_row = []
                previous_change_row = []

                for to_node in range(to_size):
                    random_weight = (
                        self.random_generator.uniform(
                            -0.5,
                            0.5
                        )
                    )

                    weight_row.append(random_weight)
                    previous_change_row.append(0.0)

                weight_matrix.append(weight_row)

                previous_change_matrix.append(
                    previous_change_row
                )

            # สร้าง Bias สำหรับ Node ในชั้นถัดไป
            bias_vector = []
            previous_bias_vector = []

            for to_node in range(to_size):
                random_bias = (
                    self.random_generator.uniform(
                        -0.5,
                        0.5
                    )
                )

                bias_vector.append(random_bias)
                previous_bias_vector.append(0.0)

            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

            self.previous_weight_changes.append(
                previous_change_matrix
            )

            self.previous_bias_changes.append(
                previous_bias_vector
            )

        # ใช้เก็บ Output ของทุก Layer
        # จาก Forward propagation รอบล่าสุด
        self.last_activations = None

    def sigmoid(self, value):
        """
        Sigmoid Activation Function

        ผลลัพธ์จะอยู่ในช่วง 0 ถึง 1
        """

        # ป้องกันเลขยกกำลังมีค่ามากเกินไป
        if value <= -60:
            return 0.0

        if value >= 60:
            return 1.0

        return 1.0 / (
            1.0 + E ** (-value)
        )

    def sigmoid_derivative(
        self,
        sigmoid_output
    ):
        """
        อนุพันธ์ของ Sigmoid

        รับค่าที่ผ่าน Sigmoid มาแล้ว
        """

        return (
            sigmoid_output
            * (1.0 - sigmoid_output)
        )

    def get_architecture(self):
        """
        คืนข้อความแสดงโครงสร้าง Network
        เช่น 8 -> 8 -> 4 -> 1
        """

        parts = []

        for size in self.layer_sizes:
            parts.append(str(size))

        return " -> ".join(parts)

    def forward(self, inputs):
        """
        ทำ Forward propagation ผ่านทุก Layer

        คืนค่า:
        1. Output ของ Hidden layer สุดท้าย
        2. Output สุดท้ายของ Network
        """

        if len(inputs) != self.input_size:
            raise ValueError(
                f"ต้องการ Input {self.input_size} ค่า "
                f"แต่ได้รับ {len(inputs)} ค่า"
            )

        # activations[0] คือ Input layer
        activations = [inputs.copy()]

        # วนผ่าน Weight Matrix ทุกชุด
        for layer_index in range(
            len(self.weights)
        ):
            previous_outputs = activations[-1]
            current_outputs = []

            number_of_current_nodes = len(
                self.biases[layer_index]
            )

            # คำนวณ Node ใน Layer ปัจจุบัน
            for current_node in range(
                number_of_current_nodes
            ):
                # เริ่มด้วย Bias
                total = self.biases[
                    layer_index
                ][current_node]

                # นำ Output จาก Layer ก่อนหน้า
                # คูณ Weight แล้วรวมกัน
                for previous_node in range(
                    len(previous_outputs)
                ):
                    total += (
                        previous_outputs[
                            previous_node
                        ]
                        * self.weights[
                            layer_index
                        ][previous_node][current_node]
                    )

                # นำผลรวมผ่าน Sigmoid
                current_output = self.sigmoid(
                    total
                )

                current_outputs.append(
                    current_output
                )

            activations.append(current_outputs)

        # เก็บ Output ของทุก Layer
        # สำหรับใช้ใน Backpropagation
        self.last_activations = activations

        final_outputs = activations[-1]

        # Output ของ Hidden layer สุดท้าย
        last_hidden_outputs = activations[-2]

        return last_hidden_outputs, final_outputs

    def backward(
        self,
        inputs,
        hidden_outputs,
        final_outputs,
        targets
    ):
        """
        ทำ Backpropagation
        ด้วย Generalized Delta Rule

        รองรับ Hidden layer หลายชั้น
        """

        if len(targets) != self.output_size:
            raise ValueError(
                f"ต้องการ Target {self.output_size} ค่า "
                f"แต่ได้รับ {len(targets)} ค่า"
            )

        # ปกติ train_one() จะเรียก forward() ก่อน
        if self.last_activations is None:
            self.forward(inputs)

        activations = self.last_activations

        number_of_weight_layers = len(
            self.weights
        )

        # Delta แต่ละตำแหน่งตรงกับ
        # Layer ปลายทางของ Weight แต่ละชุด
        deltas = []

        for layer_index in range(
            number_of_weight_layers
        ):
            deltas.append(None)

        # ----------------------------------
        # 1. คำนวณ Output Delta
        # ----------------------------------
        output_deltas = []

        for output_index in range(
            self.output_size
        ):
            output_value = final_outputs[
                output_index
            ]

            target_value = targets[
                output_index
            ]

            # Error = ค่าจริง - ค่าทำนาย
            error = target_value - output_value

            # Output delta
            delta = (
                error
                * self.sigmoid_derivative(
                    output_value
                )
            )

            output_deltas.append(delta)

        deltas[-1] = output_deltas

        # ----------------------------------
        # 2. คำนวณ Hidden Delta ย้อนกลับ
        # ----------------------------------
        for layer_index in range(
            number_of_weight_layers - 2,
            -1,
            -1
        ):
            current_layer_outputs = (
                activations[layer_index + 1]
            )

            next_layer_deltas = deltas[
                layer_index + 1
            ]

            next_layer_weights = self.weights[
                layer_index + 1
            ]

            current_layer_deltas = []

            for current_node in range(
                len(current_layer_outputs)
            ):
                error_from_next_layer = 0.0

                for next_node in range(
                    len(next_layer_deltas)
                ):
                    error_from_next_layer += (
                        next_layer_deltas[next_node]
                        * next_layer_weights[
                            current_node
                        ][next_node]
                    )

                current_output = (
                    current_layer_outputs[
                        current_node
                    ]
                )

                current_delta = (
                    error_from_next_layer
                    * self.sigmoid_derivative(
                        current_output
                    )
                )

                current_layer_deltas.append(
                    current_delta
                )

            deltas[layer_index] = (
                current_layer_deltas
            )

        # ----------------------------------
        # 3. ปรับ Weight และ Bias ทุก Layer
        # ----------------------------------
        for layer_index in range(
            number_of_weight_layers
        ):
            # Output จาก Layer ก่อนหน้า
            previous_outputs = activations[
                layer_index
            ]

            # Delta ของ Layer ปัจจุบัน
            current_deltas = deltas[
                layer_index
            ]

            # ปรับ Weight
            for previous_node in range(
                len(previous_outputs)
            ):
                for current_node in range(
                    len(current_deltas)
                ):
                    previous_change = (
                        self.previous_weight_changes[
                            layer_index
                        ][previous_node][current_node]
                    )

                    # Generalized Delta Rule
                    current_change = (
                        self.learning_rate
                        * current_deltas[current_node]
                        * previous_outputs[
                            previous_node
                        ]
                        + self.momentum
                        * previous_change
                    )

                    self.weights[layer_index][
                        previous_node
                    ][current_node] += (
                        current_change
                    )

                    self.previous_weight_changes[
                        layer_index
                    ][previous_node][current_node] = (
                        current_change
                    )

            # ปรับ Bias
            for current_node in range(
                len(current_deltas)
            ):
                previous_change = (
                    self.previous_bias_changes[
                        layer_index
                    ][current_node]
                )

                current_change = (
                    self.learning_rate
                    * current_deltas[current_node]
                    + self.momentum
                    * previous_change
                )

                self.biases[layer_index][
                    current_node
                ] += current_change

                self.previous_bias_changes[
                    layer_index
                ][current_node] = (
                    current_change
                )

    def train_one(self, inputs, targets):
        """
        ฝึก Network ด้วยข้อมูลหนึ่งตัวอย่าง
        """

        hidden_outputs, final_outputs = (
            self.forward(inputs)
        )

        total_squared_error = 0.0

        for output_index in range(
            self.output_size
        ):
            difference = (
                targets[output_index]
                - final_outputs[output_index]
            )

            total_squared_error += (
                difference ** 2
            )

        mse = (
            total_squared_error
            / self.output_size
        )

        self.backward(
            inputs,
            hidden_outputs,
            final_outputs,
            targets
        )

        return mse, final_outputs

    def predict(self, inputs):
        """
        ทำนายผลโดยไม่ปรับ Weight
        """

        hidden_outputs, final_outputs = (
            self.forward(inputs)
        )

        return final_outputs

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
        """
        ฝึก Network ด้วย Training data ทั้งหมด

        หนึ่ง Epoch คือการใช้ Training data
        ครบทุกตัวอย่างหนึ่งรอบ
        """

        if len(training_inputs) != len(
            training_targets
        ):
            raise ValueError(
                "จำนวน Input และ Target ไม่เท่ากัน"
            )

        if len(training_inputs) == 0:
            raise ValueError(
                "ไม่มี Training data"
            )

        shuffle_generator = SimpleRandom(seed)

        indices = list(
            range(len(training_inputs))
        )

        error_history = []

        for epoch in range(
            1,
            max_epochs + 1
        ):
            # สลับลำดับข้อมูลในแต่ละ Epoch
            if shuffle:
                shuffle_generator.shuffle(indices)

            total_mse = 0.0

            for data_index in indices:
                mse, outputs = self.train_one(
                    training_inputs[data_index],
                    training_targets[data_index]
                )

                total_mse += mse

            average_mse = (
                total_mse
                / len(training_inputs)
            )

            error_history.append(average_mse)

            if (
                epoch == 1
                or epoch % report_every == 0
                or epoch == max_epochs
            ):
                print(
                    f"Epoch {epoch:5d} | "
                    f"Training MSE = "
                    f"{average_mse:.8f}"
                )

            # หยุดก่อนครบ Epoch ได้
            # ถ้า Error ต่ำพอแล้ว
            if (
                error_threshold is not None
                and average_mse
                <= error_threshold
            ):
                print(
                    "หยุดก่อนครบ Epoch เพราะ "
                    "MSE ต่ำกว่า Threshold"
                )

                break

        return error_history