from pathlib import Path

# date path root
DATA_HEAD = Path("data")

# columns which are just a number, not string or object (float, int)
NUMERIC_COLS = ['Year', 'Month', 'MainLandfallLocation', 'Flood', 'Slide', 'OFDAResponse', 'Appeal', 'Declaration', 'LandfallMagnitude(kph)','LandfallPressure(mb)', 'TotalDeaths', 'NoInjured','TotalDamage(000US$)', 'TotalDamageAdjusted(000US$)', 'CPI']

# columns which values are 0 or 1, for categorical tasks
CATEGORICAL_COLS  = ['Flood', 'Slide', 'OFDAResponse', 'Appeal', 'Declaration']

# columns which have linear correlation value with each other
LINEAR_NUMERICALS_COLS = ['LandfallMagnitude(kph)', 'LandfallPressure(mb)', 'TotalDeaths', 'NoInjured', 'TotalDamage(000US$)', 'TotalDamageAdjusted(000US$)']

# linear targets
LINEAR_TARGETS = ["TotalDeaths", "NoInjured", "TotalDamageAdjusted(000US$)"]

# categorical target
CATEGORICAL_TARGETS = ['Flood', 'Slide']