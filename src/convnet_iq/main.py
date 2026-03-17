from convnet_iq.iq_generation.channel import add_noise, add_interference, apply_low_pass_filter
from convnet_iq.iq_generation.maps import index_to_symbol


def main():
    from .iq_generation.IQGenerator import IQGenerator
    gen = IQGenerator(seed=13)
    # iq_data = gen.generate_signals(n_samples=5, length=10, modulation_scheme="QPSK")
    # # print("Generated QPSK IQ data (shape: {}):".format(iq_data.shape))
    # # print(iq_data)
    # labels = gen.generate_softmax_indices_for_signals(iq_data, "QPSK")
    # print(index_to_symbol[labels])
    dataset = gen.generate_dataset()
    # interference = gen.generate_dataset(seed=42)
    print(dataset.data)
    print("------------------------------------------")
    print(dataset.labels)
    noisy_data = apply_low_pass_filter(dataset)
    print(noisy_data.data[0])


if __name__ == "__main__":
    main()
