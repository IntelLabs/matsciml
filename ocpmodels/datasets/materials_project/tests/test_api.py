
from ocpmodels.datasets.materials_project import MaterialsProjectRequest

def test_devset():
    dev_request = MaterialsProjectRequest.devset()
    data = dev_request.retrieve_data()
    import pdb; pdb.set_trace()
