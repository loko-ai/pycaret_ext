import pycaret.datasets as datasets
import pycaret.classification as pc

data = datasets.get_data('diabetes')

print(data.columns[-1])
