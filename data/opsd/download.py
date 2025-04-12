import requests
import json

import numpy as np
import pandas as pd


OPSD_URL = 'https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv'
energy_target_mapping = {
    'load_actual_entsoe_transparency': 'load',
    'solar_generation_actual': 'solar',
    'wind_generation_actual': 'wind',
    'wind_onshore_generation_actual': 'wind_onshore',
    'wind_offshore_generation_actual': 'wind_offshore',
    # capacity keys
    'solar_capacity': 'solar_capacity',
    'wind_capacity': 'wind_capacity',
    'wind_onshore_capacity': 'wind_onshore_capacity',
    'wind_offshore_capacity': 'wind_offshore_capacity',
}


def download(url, output):
    r = requests.get(url)
    with open(output, "wb") as file:
        file.write(r.content)



def load_csv(path):
    with open(path, 'rb') as file:
        opsd = pd.read_csv(file)
        file.close()
    opsd['utc_timestamp'] = pd.to_datetime(opsd['utc_timestamp'])
    opsd.set_index('utc_timestamp', inplace=True)
    return opsd


def resample(opsd):
    for key in opsd:
        opsd[key].resample()


def main():
    # download and load original opsd file
    download(OPSD_URL, 'opsd_orig.csv')
    opsd = load_csv('opsd_orig.csv')

    # drop not needed
    opsd = opsd.drop(columns=['cet_cest_timestamp'])
    for key in ['forecast', 'price', 'profile']:
        keep_columns = [x for x in opsd.columns if key not in x]
        opsd = opsd.loc[:, keep_columns]

    # generate column index
    columns_mapper = {}
    opsd.columns = [x.replace('GB_UKM', 'GB') for x in opsd.columns]
    for column in opsd.columns:
        split = column.split('_')
        region = split[0].lower()
        energy_target_name = '_'.join(split[1:])
        if energy_target_name in energy_target_mapping.keys():
            # region is a country
            energy_target = energy_target_mapping[energy_target_name]
        else:
            # region is a subregion of a country
            region = f'{region}_{split[1]}'
            energy_target_name = '_'.join(split[2:])
            energy_target = energy_target_mapping[energy_target_name]
        columns_mapper[column] = (region, energy_target)
    opsd.columns = pd.MultiIndex.from_tuples(columns_mapper.values())

    # calculate overall EU load, solar, and wind
    energy_targets = [
        'load',
        'solar',
        'wind_onshore',
        'wind_offshore'
    ]
    selected_countries = {}
    for energy_target in energy_targets:
        countries = np.unique([region for (region, target) in opsd.columns
                               if '_' not in region and target == energy_target])
        countries = np.sort(countries)
        selected_countries[energy_target] = []
        for country in countries:
            energy_series = opsd.loc[:, (country, energy_target)]
            filter = np.where(~energy_series.isna())
            starting_2015 = energy_series.index[filter[0][0]].tz_localize(None) <= pd.to_datetime('2015-01-14')
            no_missing_values = max([energy_series.loc[str(year)].isna().sum() for year in range(2015, 2020)]) < 24 * 5
            if len(filter) != 0 and starting_2015 and no_missing_values:
                selected_countries[energy_target].append(country)
        energy_data = opsd.loc[:, (selected_countries[energy_target], energy_target)].sum(axis=1)
        opsd.loc[:, ('eu', energy_target)] = energy_data

    with open('eu_countries.json', 'w') as json_file:
        json.dump(selected_countries, json_file)

    regions = [region for region, _ in opsd.columns]
    for region in regions:
        columns = opsd[region].columns
        keys = [col for col in columns
                if col == 'wind_onshore'
                or col == 'wind_offshore']
        if 'wind' not in columns and len(keys) > 0:
            opsd.loc[:, (region, 'wind')] = opsd.loc[:, (region, keys)].sum(axis=1)

    # shift opsd data by 1h to match weather forecast data
    opsd.index = opsd.index + pd.to_timedelta('1h')

    # save opsd csv file
    opsd.to_csv('opsd.csv')

if __name__ == '__main__':
    main()
