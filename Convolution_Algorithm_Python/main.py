def convolution(x, h):
    """
    Performs convolution and returns y[n] = x[n]*h[n]
    """
    output_size = len(x) + len(h) - 1
    y = [0 for _ in range(output_size)]

    for i in range(0, output_size):
        j = i
        while j >= 0:
            if(len(h) > (i - j) and len(x) > j):
                y[i] += x[j]*h[i-j]
            else:
                y[i] += 0
            j -= 1

    return y


if __name__ == "__main__":
    x = [2,4,6,4,2]
    h = [3,-1,2,1]

    print(convolution(x,h))