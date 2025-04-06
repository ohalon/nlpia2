import pandas as pd
import requests

url = 'https://www.ethnologue.com/sites/default/files/LanguageCodes.tab'

# urllib can't GET the page, but requests.get has not problem:
# pd.read_csv(url, sep='\t')

r = requests.get(url)
data = [line.split('\t') for line in r.text.splitlines()]
columns = data[0]
df = pd.DataFrame(data[1:], columns=columns)
df
#      LangID CountryID LangStatus                 Name
# 0       aaa        NG          L               Ghotuo
# 1       aab        NG          L           Alumu-Tesu
# 2       aac        PG          L                  Ari
# 3       aad        PG          L                 Amal
# 4       aae        IT          L  Albanian, Arbëreshë
# ...     ...       ...        ...                  ...
# 7481    zyg        CN          L         Zhuang, Yang
# 7482    zyj        CN          L     Zhuang, Youjiang
# 7483    zyn        CN          L      Zhuang, Yongnan
# 7484    zyp        MM          L          Chin, Zyphe
# 7485    zzj        CN          L     Zhuang, Zuojiang

# [7486 rows x 4 columns]

