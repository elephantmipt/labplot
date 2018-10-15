# This Python file uses the following encoding: utf-8


def leastsqr(x, y):
    import numpy as np
    b = (np.mean(x * y) - np.mean(x) * np.mean(y)) /\
        (np.mean(x * x) - np.mean(x) ** 2)
    a = np.mean(y) - b * np.mean(x)
    sig_b = np.sqrt(((np.mean(y ** 2) - np.mean(y) ** 2) /
                    ((np.mean(x ** 2) - np.mean(x) ** 2)) - b ** 2)/len(x))
    sig_a = sig_b * (np.mean(x * x) - np.mean(x) ** 2) ** 0.5
    return b, a, sig_b, sig_a


def make_plots(x_array, y_array, labels, err_y=0, title='', x_label='',
               style='different', y_label='', grid=False, method='polynomial',
               degree=1):
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.interpolate import CubicSpline
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import TheilSenRegressor
    from matplotlib import cm
    fig, ax = plt.subplots(figsize=(12, 8))
    if style == 'blue':
        colors_points = cm.autumn(np.linspace(0, 1, len(x_array)))
        colors_plots = ['b' for i in range(len(x_array))]
    if style == 'different':
        colors_points = cm.tab10(np.linspace(0, 0.49, len(x_array)))
        colors_plots = cm.tab10(np.linspace(0.5, 1, len(x_array)))
    for x, y, c_point, l, c_plot in zip(x_array, y_array, colors_points,
                                        labels, colors_plots):
        x_plot = np.linspace(min(x), max(x), 200)
        ax.errorbar(x, y, fmt='o', yerr=err_y, label=l, color=c_point)
        plt.legend()
        if method == 'cubic':
            smth = CubicSpline(x, y)
            y_plot = smth(x_plot)
        elif method == 'polynomial_r':
            x_pol = x[:, np.newaxis]
            degree = int(degree)
            import warnings
            warnings.filterwarnings(action="ignore", module="scipy",
                                    message="^internal gelsd")
            smth = make_pipeline(PolynomialFeatures(degree),
                                 Ridge(alpha=1*10**-16))
            smth.fit(x_pol, y)
            y_plot = smth.predict(x_plot[:, np.newaxis])
        elif method == 'polynomial_t':
            x_pol = x[:, np.newaxis]
            import warnings
            warnings.filterwarnings(action="ignore", module="scipy",
                                    message="^internal gelsd")
            smth = make_pipeline(PolynomialFeatures(degree),
                                 TheilSenRegressor())
            smth.fit(x_pol, y)
            y_plot = smth.predict(x_plot[:, np.newaxis])
        plt.plot(x_plot, y_plot, color=c_plot)
    ax.set_title(title)
    plt.grid(grid)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def interpolate_values(x, y, method='cubic'):
    from scipy.interpolate import CubicSpline
    import numpy as np
    x_interpolated = np.linspace(min(x), max(x), 500)
    if method == 'cubic':
        smth = CubicSpline(x, y)
        y_interpolated = smth(x_interpolated)
    return x_interpolated, y_interpolated


def find_nearest(array, value):
    import numpy as np
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def make_plot(x, y, lab_x='', lab_y='', title='', err_y=None, grid=True,
              y_scale='linear', interpolate_type='RBF', degree=None,
              smooth=True, linear=True, borders=True):
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
    from scipy.interpolate import CubicSpline
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import TheilSenRegressor, Ridge
    if borders:
        delta = max(x) - min(x)
        if isinstance(borders, bool):
            x1 = np.linspace(min(x) - 0.1 * delta, max(x) + 0.1 * delta, 400)
        elif isinstance(borders, float):
            x1 = np.linspace(min(x) - borders * delta,
                             max(x) + 0.1 * borders, 400)
    else:
        x1 = np.linspace(min(x), max(x), 400)
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.xlabel(lab_x)
    plt.ylabel(lab_y)
    plt.grid(grid)
    ax.set_yscale(y_scale)
    ax.errorbar(x, y, fmt='.', yerr=err_y, color='r', label=u'Данные')
    if smooth:
        X_plot = x1[:, np.newaxis]
        flag_model = False
        if interpolate_type == 'RBF':
            smth = Rbf(x, y)
        elif interpolate_type == 'IUS':
            smth = InterpolatedUnivariateSpline(x, y)
        elif interpolate_type == 'cubic':
            smth = CubicSpline(x, y)
        elif interpolate_type == 'polynomial_r':
            X = x[:, np.newaxis]
            degree = int(degree)
            import warnings
            warnings.filterwarnings(action="ignore", module="scipy",
                                    message="^internal gelsd")
            smth = make_pipeline(PolynomialFeatures(degree),
                                 Ridge(alpha=1*10**-16))
            smth.fit(X, y)
            flag_model = True
        elif interpolate_type == 'polynomial_t':
            X = x[:, np.newaxis]
            degree = int(degree)
            import warnings
            warnings.filterwarnings(action="ignore", module="scipy",
                                    message="^internal gelsd")
            smth = make_pipeline(PolynomialFeatures(degree),
                                 TheilSenRegressor())
            smth.fit(X, y)
            flag_model = True
        else:
            raise Exception("Invalid interpolate type: You can choose\
                   RBF, IUS, cubic for cubic spline , polynomial_r for Ridge\
                   regressor polynomial_t for TheilSen regressor")
        if flag_model:
            y1 = smth.predict(X_plot)
        else:
            y1 = smth(x1)
        plt.plot(x1, y1, color='g', label=u'Сглаживание')
    b1, a1, sig_b1, sig_a1 = leastsqr(x, y)
    label_ = u'\nb={}±{}  a={}±{}'.format(np.round(b1, 3), np.round(sig_b1, 3),
                                          np.round(a1, 3), np.round(sig_a1, 3))
    ax.set_title(title)
    if linear:
        plt.plot(x1, b1 * x1 + a1, label=u'Линейная апроксимация' + label_)
    plt.legend()
    return fig, ax


if __name__ == '__main__':
    docstring = 'Hello, please add this file to your main\
                 program directory and then import it :)'
    print(docstring)
