import requests
# Consumentenvertrouwen 83693NED via OData v4
# Period formaat: "2025MM11" (yyyy 'MM' month)
def get_consumer_confidence(period_code: str, dataset: str = "83693NED"):
    url = f"https://opendata.cbs.nl/ODataApi/OData/{dataset}/TypedDataSet?$filter=Periods%20eq%20%27{period_code}%27"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    val = r.json()["value"][0]["ConsumerConfidence_2"]  # veldnaam in dataset
    return {"value": val}
