def load_from_file(file_name, start_step=0):
    step_list = []
    mean_list = []
    with open(file_name, 'r') as f:
        for line in f:
            if 'mean = ' not in line:
                continue
            line = line.replace('mean = ', ',')
            line = line.replace('max = ', ',')
            line = line.replace(' ', '')
            step, mean = line.split(',')[:2]
            step_list.append(int(step)+start_step)
            mean_list.append(float(mean))
    return step_list, mean_list

def load_data():
    step_list = []
    mean_list = []
    for file in ['log_200000.txt', 'log_400000.txt']:
        step, mean = load_from_file(file, len(step_list)*1000)
        step_list += step
        mean_list += mean

    return step_list, mean_list


def plot(step_list, mean_list):
    import matplotlib.pyplot as plt
    plt.plot(step_list, mean_list)
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.savefig('plot.png')


if __name__ == '__main__':
    step_list, mean_list = load_data()
    plot(step_list, mean_list)
