# math ใช้คำนวณ exponential ใน Sigmoid
import math

# random ใช้สุ่ม Weight และ Bias เริ่มต้น
import random


class MLP:
    """
    Multi-Layer Perceptron แบบ 1 Hidden Layer

    โครงสร้าง:
    Input Layer -> Hidden Layer -> Output Layer
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        learning_rate=0.01,
        momentum=0.0,
        seed=None
    ):
        """
        กำหนดโครงสร้างและค่าพารามิเตอร์ของ Network
        """

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.learning_rate = learning_rate
        self.momentum = momentum

        # สร้างตัวสุ่มเฉพาะของ Network นี้
        # เมื่อใช้ seed เดิม จะได้ Weight เริ่มต้นเหมือนเดิม
        self.random_generator = random.Random(seed)

        # -------------------------------------------------
        # สุ่ม Weight ระหว่าง Input Layer กับ Hidden Layer
        # ขนาด input_size x hidden_size
        # -------------------------------------------------
        self.weights_input_hidden = []

        for input_index in range(input_size):
            weight_row = []

            for hidden_index in range(hidden_size):
                random_weight = self.random_generator.uniform(
                    -0.5,
                    0.5
                )
                weight_row.append(random_weight)

            self.weights_input_hidden.append(weight_row)

        # Bias ของ Hidden Layer
        self.bias_hidden = []

        for hidden_index in range(hidden_size):
            random_bias = self.random_generator.uniform(
                -0.5,
                0.5
            )
            self.bias_hidden.append(random_bias)

        # -------------------------------------------------
        # สุ่ม Weight ระหว่าง Hidden Layer กับ Output Layer
        # ขนาด hidden_size x output_size
        # -------------------------------------------------
        self.weights_hidden_output = []

        for hidden_index in range(hidden_size):
            weight_row = []

            for output_index in range(output_size):
                random_weight = self.random_generator.uniform(
                    -0.5,
                    0.5
                )
                weight_row.append(random_weight)

            self.weights_hidden_output.append(weight_row)

        # Bias ของ Output Layer
        self.bias_output = []

        for output_index in range(output_size):
            random_bias = self.random_generator.uniform(
                -0.5,
                0.5
            )
            self.bias_output.append(random_bias)

        # -------------------------------------------------
        # เก็บค่าเปลี่ยนแปลง Weight จากรอบก่อน
        # ใช้สำหรับคำนวณ Momentum
        # -------------------------------------------------

        self.previous_change_input_hidden = []

        for input_index in range(input_size):
            change_row = []

            for hidden_index in range(hidden_size):
                change_row.append(0.0)

            self.previous_change_input_hidden.append(change_row)

        self.previous_change_hidden_output = []

        for hidden_index in range(hidden_size):
            change_row = []

            for output_index in range(output_size):
                change_row.append(0.0)

            self.previous_change_hidden_output.append(change_row)

        # ค่าเปลี่ยนแปลง Bias จากรอบก่อน
        self.previous_change_bias_hidden = [
            0.0 for _ in range(hidden_size)
        ]

        self.previous_change_bias_output = [
            0.0 for _ in range(output_size)
        ]

    def sigmoid(self, value):
        """
        Sigmoid Activation Function

        ผลลัพธ์อยู่ในช่วง 0 ถึง 1
        """

        # ป้องกันค่า exponential ใหญ่เกินไป
        if value <= -60:
            return 0.0

        if value >= 60:
            return 1.0

        return 1.0 / (1.0 + math.exp(-value))

    def sigmoid_derivative(self, sigmoid_output):
        """
        อนุพันธ์ของ Sigmoid

        ฟังก์ชันนี้รับค่าที่ผ่าน Sigmoid มาแล้ว
        """
        return sigmoid_output * (1.0 - sigmoid_output)

    def forward(self, inputs):
        """
        ทำ Forward Propagation

        คืนค่า:
        1. ผลลัพธ์ของ Hidden Layer
        2. ผลลัพธ์สุดท้ายของ Output Layer
        """

        if len(inputs) != self.input_size:
            raise ValueError(
                f"Network ต้องการ Input {self.input_size} ค่า "
                f"แต่ได้รับ {len(inputs)} ค่า"
            )

        # ------------------------------------
        # Input Layer -> Hidden Layer
        # ------------------------------------
        hidden_outputs = []

        for hidden_index in range(self.hidden_size):
            total = self.bias_hidden[hidden_index]

            for input_index in range(self.input_size):
                input_value = inputs[input_index]

                weight = self.weights_input_hidden[
                    input_index
                ][hidden_index]

                total += input_value * weight

            hidden_output = self.sigmoid(total)
            hidden_outputs.append(hidden_output)

        # ------------------------------------
        # Hidden Layer -> Output Layer
        # ------------------------------------
        final_outputs = []

        for output_index in range(self.output_size):
            total = self.bias_output[output_index]

            for hidden_index in range(self.hidden_size):
                hidden_value = hidden_outputs[hidden_index]

                weight = self.weights_hidden_output[
                    hidden_index
                ][output_index]

                total += hidden_value * weight

            final_output = self.sigmoid(total)
            final_outputs.append(final_output)

        return hidden_outputs, final_outputs

    def backward(
        self,
        inputs,
        hidden_outputs,
        final_outputs,
        targets
    ):
        """
        ทำ Backpropagation ด้วย Generalized Delta Rule

        ขั้นตอน:
        1. คำนวณ Output Delta
        2. คำนวณ Hidden Delta
        3. ปรับ Hidden-to-Output Weight
        4. ปรับ Input-to-Hidden Weight
        """

        if len(targets) != self.output_size:
            raise ValueError(
                f"Network ต้องการ Target {self.output_size} ค่า "
                f"แต่ได้รับ {len(targets)} ค่า"
            )

        # ------------------------------------
        # 1. คำนวณ Output Delta
        # ------------------------------------
        output_deltas = []

        for output_index in range(self.output_size):
            target = targets[output_index]
            output = final_outputs[output_index]

            # Error = ค่าจริง - ค่าทำนาย
            output_error = target - output

            # Delta = Error x อนุพันธ์ของ Activation
            output_delta = (
                output_error
                * self.sigmoid_derivative(output)
            )

            output_deltas.append(output_delta)

        # ------------------------------------
        # 2. คำนวณ Hidden Delta
        # ต้องทำก่อนปรับ Hidden-to-Output Weight
        # เพราะต้องใช้ Weight ค่าเดิม
        # ------------------------------------
        hidden_deltas = []

        for hidden_index in range(self.hidden_size):
            error_from_output = 0.0

            for output_index in range(self.output_size):
                error_from_output += (
                    output_deltas[output_index]
                    * self.weights_hidden_output[
                        hidden_index
                    ][output_index]
                )

            hidden_output = hidden_outputs[hidden_index]

            hidden_delta = (
                error_from_output
                * self.sigmoid_derivative(hidden_output)
            )

            hidden_deltas.append(hidden_delta)

        # ------------------------------------
        # 3. ปรับ Weight Hidden -> Output
        # ------------------------------------
        for hidden_index in range(self.hidden_size):
            for output_index in range(self.output_size):

                previous_change = (
                    self.previous_change_hidden_output[
                        hidden_index
                    ][output_index]
                )

                current_change = (
                    self.learning_rate
                    * output_deltas[output_index]
                    * hidden_outputs[hidden_index]
                    + self.momentum
                    * previous_change
                )

                self.weights_hidden_output[
                    hidden_index
                ][output_index] += current_change

                self.previous_change_hidden_output[
                    hidden_index
                ][output_index] = current_change

        # ปรับ Bias ของ Output Layer
        for output_index in range(self.output_size):
            previous_change = (
                self.previous_change_bias_output[
                    output_index
                ]
            )

            current_change = (
                self.learning_rate
                * output_deltas[output_index]
                + self.momentum
                * previous_change
            )

            self.bias_output[output_index] += current_change

            self.previous_change_bias_output[
                output_index
            ] = current_change

        # ------------------------------------
        # 4. ปรับ Weight Input -> Hidden
        # ------------------------------------
        for input_index in range(self.input_size):
            for hidden_index in range(self.hidden_size):

                previous_change = (
                    self.previous_change_input_hidden[
                        input_index
                    ][hidden_index]
                )

                current_change = (
                    self.learning_rate
                    * hidden_deltas[hidden_index]
                    * inputs[input_index]
                    + self.momentum
                    * previous_change
                )

                self.weights_input_hidden[
                    input_index
                ][hidden_index] += current_change

                self.previous_change_input_hidden[
                    input_index
                ][hidden_index] = current_change

        # ปรับ Bias ของ Hidden Layer
        for hidden_index in range(self.hidden_size):
            previous_change = (
                self.previous_change_bias_hidden[
                    hidden_index
                ]
            )

            current_change = (
                self.learning_rate
                * hidden_deltas[hidden_index]
                + self.momentum
                * previous_change
            )

            self.bias_hidden[hidden_index] += current_change

            self.previous_change_bias_hidden[
                hidden_index
            ] = current_change

    def train_one(self, inputs, targets):
        """
        ฝึก Network ด้วยข้อมูลหนึ่งตัวอย่าง

        คืนค่า:
        1. Mean Squared Error
        2. Output ก่อนปรับ Weight
        """

        hidden_outputs, final_outputs = self.forward(inputs)

        total_squared_error = 0.0

        for output_index in range(self.output_size):
            difference = (
                targets[output_index]
                - final_outputs[output_index]
            )

            total_squared_error += difference ** 2

        mse = total_squared_error / self.output_size

        self.backward(
            inputs,
            hidden_outputs,
            final_outputs,
            targets
        )

        return mse, final_outputs