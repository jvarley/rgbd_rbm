from pylearn2.gui import patch_viewer
from pylearn2.utils import serial

def show(model_path):

    model = serial.load(model_path)

    patch_view = patch_viewer.PatchViewer((80, 80), (8, 8))

    weights = model.get_weights()
    print weights.shape

    # for i in range(0, weights.shape[1]):
    #     for j in range(0,4):
    #         patch = weights[64*j:64*(j+1),i].reshape(8,8)
    #         patch_view.add_patch(patch)
    #
    # patch_view.show()



    patch_view = patch_viewer.PatchViewer((80, 80), (8, 8))
    for i in range(0, weights.shape[1]):
        for j in range(0,4):
            patch = weights[j::4,i].reshape(8,8)
            patch_view.add_patch(patch)

    patch_view.show()

#show('rgbd_dataset.pkl')
show('rgbd_rbm_model.pkl')
