def main():
    from .iq_generation.IQGenerator import IQGenerator
    gen = IQGenerator(seed=123)
    iq_data = gen.generate_qpsk(n_samples=5, length=10)
    print("Generated QPSK IQ data (shape: {}):".format(iq_data.shape))
    print(iq_data)


if __name__ == "__main__":
    main()
