from dsa2000_fm.systematics.ionosphere import SimulationResult


def main(save_file):
    result = SimulationResult.parse_file(save_file)
    result.interactive_explore()


if __name__ == '__main__':
    main('predicted_lwa_tec.json')
