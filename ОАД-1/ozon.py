import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from scipy.io import netcdf

with netcdf.netcdf_file('/Users/akonkina/Downloads/MSR-2.nc', mmap=False) as nf:
    data = nf.variables
    ozon = data['Average_O3_column'].data
    lat = data['latitude'].data
    long = data['longitude'].data
    time = data['time'].data
    parser = argparse.ArgumentParser()
    parser.add_argument('longitude', metavar='LON', type=float, help='Longitude, deg')
    parser.add_argument('latitude', metavar='LAT', type=float, help='Latitude, deg')

    if __name__ == "__main__":
        args = parser.parse_args()
        print(args.longitude, args.latitude)

    la = np.searchsorted(lat, args.latitude)
    lo = np.searchsorted(long, args.longitude)

    z = ozon[:, la, lo]
    mean_1 = np.mean(z)
    max_1 = np.max(z)
    min_1 = np.min(z)
    y = ozon[0:len(z):12, la, lo]
    mean_2 = np.mean(y)
    max_2 = np.max(y)
    min_2 = np.min(y)
    x = ozon[6:len(z):12, la, lo]
    mean_3 = np.mean(x)
    max_3 = np.max(x)
    min_3 = np.min(x)
    year = np.fix((1970 + time / 12))
    month = (time % 12 + 1)
    day = np.ones(len(month))
    times = pd.DataFrame({'year': year,
                          'month': month, 'day': day})
    times = np.array(pd.to_datetime(times))
    df_all = pd.DataFrame({'year': times, 'ozon': z})
    df_jan = pd.DataFrame({'year': times[0:len(z):12], 'ozon': y})
    df_july = pd.DataFrame({'year': times[6:len(z):12], 'ozon': x})
    plt.plot(df_all['year'], df_all['ozon'],
             label='Зависимость содержания озона от времени для всего доступного интревала')
    plt.plot(df_jan['year'], df_jan['ozon'], label='Зависимость содержания озона от времени для  январей')
    plt.plot(df_july['year'], df_july['ozon'], label='Зависимость содержания озона от времени для  июлей')
    plt.xlabel('Период за который проходили измерения (1979-2020)')
    plt.title('Содержание озона в атмосфере')
    plt.legend()
    plt.savefig('ozon.png')
    plt.show()

    import json

    d = {
        "coordinates": [37.66, 55.77],
        "jan": {
            "min": float(min_1),
            "max": float(max_2),
            "mean": round(mean_2, 2)
        },
        "jul": {
            "min": float(min_3),
            "max": float(max_3),
            "mean": round(mean_3, 2)
        },
        "all": {
            "min": float(min_1),
            "max": float(max_1),
            "mean": round(mean_1, 2)
        }
    }

    with open('ozon.json', 'w') as f:
        json.dump(d, f)
