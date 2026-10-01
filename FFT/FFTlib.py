from collections.abc import Iterable

from numpy.typing import ArrayLike, NDArray
import numpy as np


class FFT:
    """基于迭代式 Radix-2 蝶形运算的一维、二维快速傅里叶变换。

    正向变换使用负指数旋转因子，不做归一化，频点顺序与 NumPy 一致。
    每个变换维度的长度必须是 2 的整数次幂（包括 1）。
    公共方法在独立的缓冲区中计算，不修改输入数据。
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def __reverse_bits(index: int, bit_count: int) -> int:
        """反转索引的低 bit_count 位，例如 3 位索引 001 变为 100。"""
        reversed_index = 0
        for _ in range(bit_count):
            reversed_index = (reversed_index << 1) | (index & 1)
            index >>= 1
        return reversed_index

    @staticmethod
    def __dft_merge(spectrum: list[complex], inverse=False) -> list[complex]:
        """原地合并位反转排列后的数据，并返回同一个列表。

        调用前须保证长度为 2 的整数次幂，且已完成位反转排列。
        每层将两个较小的 DFT 合并，直到得到完整频谱。
        """
        sample_count = len(spectrum)
        size = 2

        sign = 1 if inverse else -1

        while size <= sample_count:
            half_size = size // 2

            # 当前层的旋转因子，同一层的所有分组共用。
            twiddle_factors = [
                np.exp(sign * 2j * np.pi * k / size)
                for k in range(half_size)
            ]

            for start in range(0, sample_count, size):
                # 配对分组前后两半中相同偏移位置的系数。
                for k in range(half_size):
                    left_index = start + k
                    right_index = left_index + half_size

                    # 保存两个输入，避免写回数组时覆盖旧值。
                    even_value = spectrum[left_index]
                    odd_value = spectrum[right_index]

                    # 蝶形运算：一次乘法，两个输出共同使用。
                    weighted_odd = twiddle_factors[k] * odd_value

                    spectrum[left_index] = even_value + weighted_odd
                    spectrum[right_index] = even_value - weighted_odd

            size *= 2
        return spectrum

    @staticmethod
    def fft(x: Iterable[complex]) -> list[complex]:
        """计算一维离散傅里叶变换。

        Args:
            x: 实数或复数采样值组成的可迭代对象，长度须为 2 的整数次幂。

        Returns:
            与输入等长的复数频谱列表，索引 k 对应第 k 个频点，零频在首位。

        Raises:
            ValueError: 输入为空或长度不是 2 的整数次幂。
        """
        # 复制输入并转为复数，后续重排和蝶形运算均在此缓冲区完成。
        sample = [complex(value) for value in x]
        sample_count = len(sample)

        # 正整数是 2 的幂，当且仅当其二进制表示中只有一个 1。
        if sample_count == 0 or (sample_count & (sample_count - 1)) != 0:
            raise ValueError("采样点数量必须是 2 的整数次幂")

        # sample_count = 2 ** bit_count，索引范围恰好占用 bit_count 位。
        bit_count = sample_count.bit_length() - 1

        # 按照位反转关系交换数组元素。
        for i in range(sample_count):
            j = FFT.__reverse_bits(i, bit_count)
            if i < j:
                sample[i], sample[j] = sample[j],sample[i]


        # 排列完成后，进行逐层 DFT 合并。
        return FFT.__dft_merge(sample)


    @staticmethod
    def fft2(image: ArrayLike) -> NDArray[np.complex128]:
        """计算二维离散傅里叶变换：先逐行变换，再逐列变换。

        Args:
            image: 可转换为复数数组的非空二维数据，宽和高均须为 2 的整数次幂。

        Returns:
            与输入形状相同的复数频谱数组。结果 [v, u] 对应纵向频点 v
            和横向频点 u，零频位于 [0, 0]。

        Raises:
            ValueError: 输入不是非空二维数组，或宽、高不是 2 的整数次幂。
        """
        values = np.asarray(image, dtype=complex)
        if values.ndim != 2 or values.size == 0:
            raise ValueError("输入必须是非空二维数组")
        height, width = values.shape
        if (height & (height - 1)) or (width & (width - 1)):
            raise ValueError("宽和高都必须为 2 的幂，请先补零")

        # 行变换后，第一个维度仍为空间位置，第二个维度已是横向频率。
        row_spectrum = np.empty((height, width), dtype=complex)
        for row_index in range(height):
            row_spectrum[row_index, :] = FFT.fft(values[row_index, :])

        # 对每个横向频点逐列变换，将纵向空间位置也转换为频率。
        spectrum = np.empty_like(row_spectrum)
        for column_index in range(width):
            spectrum[:, column_index] = FFT.fft(row_spectrum[:, column_index])
        return spectrum



#---------------------测试代码----------------------
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # 创建离散定义域列表，即采样点列表
    # 在 [0, 1) 范围内，等间隔取 1024 个点
    count = 1024
    start = 0.0
    end = 1.0
    x = np.linspace(start, end, count, endpoint=False)

    # 将全部采样点代入函数，得到长度为 1024 的离散函数结果列表
    samples = (
            np.sin(2 * np.pi * 10 * x)
            + np.sin(2 * np.pi * 200 * x)
            + np.sin(2 * np.pi * 300 * x)
    )

    # 计算FFT
    result = FFT.fft(samples)

    # FFT结果并取复数模长。并转换为numpy数组
    magnitude = np.abs(np.asarray(result, dtype=complex))
    k = np.arange(magnitude.size)

    # 显示图像。
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(x, samples)
    axes[0].set(xlabel="x", ylabel="f(x)", title="Original Signal")
    axes[0].grid(alpha=0.3)
    axes[1].plot(k, magnitude)
    axes[1].set(xlabel="Frequency bin k", ylabel="|X[k]|", title="FFT: Magnitude")
    axes[1].grid(alpha=0.3)

    output_dir = Path(__file__).resolve().parent
    fig.savefig(output_dir / "fft_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
