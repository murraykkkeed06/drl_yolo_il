import numpy as np
from darkflow.net.build import TFNet

# load the pretrained model and weight
options = {"pbLoad": "built_graph/tiny-yolo-voc-new.pb", "metaLoad": "built_graph/tiny-yolo-voc-new.meta", "threshold": 0.1, "gpu": 0.8}
tfnet = TFNet(options)

def predict_and_process(image):

    frame = np.array(img)
    result = tfnet.return_predict(frame)

    if len(result) > 0:

        conf_list = []
        for box in result:
            conf_list.append(box["confidence"])

        conf_arr = np.array(conf_list)
       
        max_id = np.argmax(conf_arr)

        print(result[max_id])

        x1, y1 = result[max_id]["topleft"]["x"], result[max_id]["topleft"]["y"]
        x2, y2 = result[max_id]["bottomright"]["x"], result[max_id]["bottomright"]["y"]
        r = np.array([y1, y1, y2, y2])
        c = np.array([x1, x2, x2, x1])
        rr, cc = polygon_perimeter(r, c)
        frame[:, :] = [0, 0, 0]
        frame[rr, cc] = [255, 255, 255]
    
    else:
        frame[:, :] = [0, 0, 0]

