

def fda_outliers(self, freq):
    '''
    :param freq: amount of values in a hour (value to divide 60 and the result is the amount of data in a hour)
    :return: the variable y with missing value in the days considered as outliers
    '''
    step = int(60 / freq)
    y = self.data.iloc[:, self.pos_y]
    long = len(y)
    hour = self.times.hour
    start = np.where(hour == 0)[0][0]

    if np.where(hour == 0)[0][len(np.where(hour == 0)[0]) - 1] > np.where(hour == 23)[0][
        len(np.where(hour == 23)[0]) - 1]:
        d = np.where(hour == 0)[0][len(np.where(hour == 0)[0]) - 1] - np.where(hour == 23)[0][
            len(np.where(hour == 23)[0]) - 1]
        end = np.where(hour == 0)[0][len(np.where(hour == 0)[0]) - 1 - d]
    elif np.where(hour == 0)[0][len(np.where(hour == 0)[0]) - 1] < np.where(hour == 23)[0][
        len(np.where(hour == 23)[0]) - 1]:
        if np.sum(hour[np.where(hour == 0)[0][len(np.where(hour == 0)[0]) - 1]:np.where(hour == 23)[0][
            len(np.where(hour == 23)[0]) - 1]] == 23) == step:
            end = np.where(hour == 23)[0][len(np.where(hour == 23)[0]) - 1]
        else:
            d = np.where(hour == 0)[0][len(np.where(hour == 0)[0]) - 1] - np.where(hour == 23)[0][
                len(np.where(hour == 23)[0]) - 1]
            end = np.where(hour == 0)[0][len(np.where(hour == 0)[0]) - 1 - d]
    else:
        end = []
        raise NameError('Problem with the limit of sample creating the functional sample')

    y1 = y.iloc[range(start + 1)]
    y2 = y.iloc[range(end - 1, len(y))]

    y_short = y.iloc[range(start + 1, end - 1)]
    if len(y_short) % (step * 24) != 0:
        print(len(y_short))
        print(len(y_short) / (step * 24))
        raise NameError('Sample size not it is well divided among days')

    fd_y = DL.cortes(y_short, len(y_short), int(24 * step)).transpose()
    print(fd_y.shape)
    grid = []
    for t in range(int(24 * step)):
        grid.append(t)

    fd_y2 = fd_y.copy()
    missing = []
    missing_p = []
    for t in range(fd_y.shape[0]):
        if np.sum(np.isnan(fd_y[t, :])) > 0:
            missing.append(t)
            missing_p.append(np.where(np.isnan(fd_y[t, :]))[0])

    if len(missing) > 0:
        fd_y3 = pd.DataFrame(fd_y2.copy())
        for j in range(len(missing)):
            fd_y3.iloc[missing[j], missing_p[j]] = self.mask_value
            fd_y2[missing[j], missing_p[j]] = self.mask_value
        index2 = fd_y3.index
        print(missing)
        print(index2)
    else:
        fd_y3 = pd.DataFrame(fd_y2.copy())
        index2 = fd_y3.index

    fd = fd_y2.tolist()
    fd1 = skfda.FDataGrid(fd, grid)

    out_detector1 = skfda.exploratory.outliers.IQROutlierDetector(factor=3,
                                                                  depth_method=skfda.exploratory.depth.BandDepth())  # MSPlotOutlierDetector()
    out_detector2 = skfda.exploratory.outliers.LocalOutlierFactor(n_neighbors=int(fd_y2.shape[0] / 5))
    oo1 = out_detector1.fit_predict(fd1)
    oo2 = out_detector2.fit_predict(fd1)
    o1 = np.where(oo1 == -1)[0]
    o2 = np.where(oo2 == -1)[0]
    o_final = np.intersect1d(o1, o2)

    print('El número de outliers detectado es:', len(o_final))
    if len(o_final) > 0:
        out = index2[o_final]

        for t in range(len(o_final)):
            w = np.empty(fd_y.shape[1])
            w[:] = self.mask_value
            fd_y[out[t], :] = w

    Y = fd_y.flatten()

    Y = pd.concat([pd.Series(y1), pd.Series(Y), pd.Series(y2)], axis=0)
    if len(Y) != long:
        print(len(Y))
        print(long)
        raise NameError('Sample size error in the second joint')

    Y.index = self.data.index
    self.data.iloc[:, self.pos_y] = Y

    print('Data have been modified masking the outliers days!')
