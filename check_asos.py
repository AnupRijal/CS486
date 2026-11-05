import requests, pandas as pd
from io import StringIO

test = ["AZA","CLD","FCA","GUF","HHH","MQT","PBI","SCE","USA","YUM"]

params = ([("station", a) for a in test]
          + [("data", "tmpf"), ("data", "vsby")]
          + [("sts", "2023-06-01T00:00:00Z"), ("ets", "2023-06-02T00:00:00Z"),
             ("tz", "UTC"), ("format", "onlycomma"), ("direct", "yes")])

r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
                 params=params, timeout=300)
d = pd.read_csv(StringIO(r.text), comment="#")

found = set(d.station.unique()) if len(d) else set()
print("found  :", sorted(found))
print("missing:", sorted(set(test) - found))