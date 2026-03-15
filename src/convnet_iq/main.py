def main():
    from .iq_generation.IQGenerator import IQGenerator
    gen = IQGenerator(seed=123)
    iq_data = gen.generate_signals(n_samples=5, length=10, modulation_scheme="QPSK")
    print("Generated QPSK IQ data (shape: {}):".format(iq_data.shape))
    print(iq_data)
    labels = gen.generate_softmax_indices_for_signals(iq_data, "QPSK")
    print(labels)

if __name__ == "__main__":
    main()
