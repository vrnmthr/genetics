import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import xlsxwriter
import powerlaw as pl

NODES = 5
T_ITERS = 7000
LOGGING_FREQUENCY = 5

vals = [5*i for i in range(4,15)]

def process(v):
    # process("{}.txt".format(file))
    f = open("./results/{}/{}raw.txt".format(NODES, v))
    reader = csv.reader(f)
    length = T_ITERS/LOGGING_FREQUENCY
    avg = np.zeros(int(length) - 1)
    sols = np.zeros(int(length) - 1)

    rows = 1
    # graphs the percentage difference
    for row in reader:
        solution = float(row[0])
        for i in range(1, len(row) - 1):
            val = float(row[i])
            pdif = (val - solution)/solution
            if rows > 1:
                avg[i-1] = (avg[i-1] * (rows - 1) + val)/rows
                sols[i - 1] = (sols[i - 1] * (rows - 1) + pdif) / rows
            else:
                avg[i-1] = val
                sols[i - 1] = pdif
        rows += 1

    #plt.plot(avg)
    np.save("./results/{}/base{}".format(NODES, v), sols)
    # plt.plot(sols, label="{}".format(v))

def graph_all():
    for v in vals:
        print("Processing {}".format(v))
        sols = np.load("./results/{}/base{}.npy".format(NODES, v))
        plt.plot(np.arange(0, T_ITERS - LOGGING_FREQUENCY, LOGGING_FREQUENCY), sols, label="{}".format(v))

    axes = plt.gca()
    axes.set_ylim([0, 2.0])
    plt.legend()
    plt.savefig("./results/{}/pdifs.png".format(NODES), dpi=200)

def fit_curves_scipy():
    f = open("./results/{}/curves.txt".format(NODES), "w+")
    for v in vals:
        print("Processing {}".format(v))
        y = np.load("./results/{}/base{}.npy".format(NODES, v))
        x = np.linspace(1,T_ITERS, T_ITERS/LOGGING_FREQUENCY - 1)
        #popt, pcov = opt.curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c, x,y, p0=(5000,-0.003,0.3))
        #popt, pcov = opt.curve_fit(lambda t, a, b: a + b * np.log(t), x, y, p0=(6,-0.5))
        popt, pcov = opt.curve_fit(lambda t, a, b: a*np.power(t,b), x, y, p0=(80, -0.5))
        perr = np.sqrt(np.diag(pcov))
        f.write("{}: y=a + b*ln(x), a = {}, b = {}, | stderr = {}\n".format(v, popt[0], popt[1], perr))
    f.close()

def fit_curves_powerlaw():
    f = open("./results/{}/curves.txt".format(NODES), "w+")
    dists = ["power_law", "truncated_power_law", "lognormal", "exponential"]
    fig = None
    for v in vals:
        print("Processing {}".format(v))
        y = np.load("./results/{}/base{}.npy".format(NODES, v))

        fit = pl.Fit(y)

        # find the best fitting distribution
        bestFit = dists[0]
        for i in range(1, len(dists)):
            R, _ = fit.distribution_compare(bestFit, dists[i])
            # if R is negative, dists[i] is better
            if R < 0:
                bestFit = dists[i]

        # build a graph of the distributions and fits
        print(bestFit)
        f.write("base = {}: ".format(v))
        tpl = fit.truncated_power_law
        f.write("alpha = {}, xmin = {}, lam = {}\n".format(tpl.alpha, tpl.xmin, tpl.Lambda))

        if fig == None:
            fig = fit.plot_pdf(linewidth=2, label="{}".format(v))
        else:
            fit.plot_pdf(linewidth=2, label="{}".format(v), ax=fig)
        fit.truncated_power_law.plot_pdf(linestyle='--', ax=fig, label="best-fit {}".format(v))

    plt.legend()
    plt.savefig("./results/{}/distributions.png".format(NODES))
    f.close()

def write_to_excel():
    workbook = xlsxwriter.Workbook("./results/{}/data.xslx".format(NODES))
    for v in vals:
        wsheet = workbook.add_worksheet("{}".format(v))
        print("Processing {}".format(v))
        y = np.load("./results/{}/base{}.npy".format(NODES, v))
        x = np.linspace(1, T_ITERS, T_ITERS / LOGGING_FREQUENCY - 1)
        r = 0
        for a,b in zip(x,y):
            wsheet.write(r, 0, a)
            wsheet.write(r, 1, b)
            r += 1
    workbook.close()


# for v in vals:
#     print("Processing {}".format(v))
#     process(v)
#
#graph_all()
fit_curves_powerlaw()
#write_to_excel()
