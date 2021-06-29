from sklearn.metrics import f1_score
from tensorflow.keras.metrics import MeanIoU

def performance_score(y_true, y_pred):
    ''' function that returns a dictionnary with the F1 and IoU scores !'''
    # Calculer score IoU 
    m=MeanIoU(num_classes=2)
    m.reset_state()
    m.update_state(y_true, y_pred)
    IoU=round(m.result().numpy(),2)
    
    # Calculer score F1    
    f1= round(f1_score(y_true=y_true, y_pred=y_pred),2)
    
    return {"F1_score":f1, "IoU_score":IoU}
