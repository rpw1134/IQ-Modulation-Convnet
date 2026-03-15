def main():
    from .iq_generation.IQGenerator import IQGenerator
    gen = IQGenerator(seed=123)
    iq_data = gen.generate_bspk(n_samples=5, length=10)
    print("Generated BPSK IQ data (shape: {}):".format(iq_data.shape))
    print(iq_data[0, 1])


if __name__ == "__main__":
    main()
