'''
Created on 2024/06/14 15:40

@Author - Dudi Levi
'''

import numpy as np

CLASS_NAMES = ["Good", "Bad", "NotRelevant"]


#==================================================================
def GetPredictionClassResults(predictions, className):
  """
  This function return the prediction value for the target class name.

  Args:
      predictions: An array containing the predicted class probabilities.
      class_names: A list containing the class names corresponding to the predictions.
      target_class_name: The class name you want to check for.

  Returns:
      None: if class was not found.
      predicurion value: if class is found.
  """
  # Create a mask for the target class
  #mask = np.array([name == target_class_name for name in CLASS_NAMES])
  mask = np.asarray(CLASS_NAMES) == target_class_name
  # Check if the target class is present
  if np.any(mask):
    # Get the index of the target class if present
    target_class_index = np.where(mask)[0][0]  # Get the first occurrence index
    return predictions[target_class_index]
  else:
    return None


#==================================================================
def GetPredictionResults(predictions):
    """ Return the high value of prediction array """
    predictedMaxValIndex = np.argmax(predictions, axis=None)  # Get the predicted class index (highest probability)
    if predictedMaxValIndex == 0:
        if np.all(predictions == predictions[0]):
            print(f"Error all prediction values are identicall to - {predictions[0]}")
            return None, None

    return CLASS_NAMES[predictedMaxValIndex], predictions[predictedMaxValIndex]



if __name__ == '__main__':

    #predictions = np.array([[0.2, 0.8, 0.1], [0.1, 0.3, 0.6]])  # Sample prediction probabilities
    predictions = np.array([0.2, 0.8, 0.1])  # Sample prediction probabilities
    target_class_name = "Good"
    res = GetPredictionClassResults(predictions, target_class_name)
    assert res == 0.2
    print(f"Target class '{target_class_name}' is {res}")
    target_class_name = "Bad"
    res = GetPredictionClassResults(predictions, target_class_name)
    assert res == 0.8
    print(f"Target class '{target_class_name}' is {res}")
    target_class_name = "koko"
    res = GetPredictionClassResults(predictions, target_class_name)
    assert res == None
    print(f"Target class '{target_class_name}' is {res} Not exists")

    predictions = np.array([0.2, 0.8, 0.1])
    className, res = GetPredictionResults(predictions)
    assert className == "Bad"
    assert res == 0.8
    print(f"Target class '{className}' is {res}")
    predictions = np.array([0.2, 0.2, 0.6])
    className, res = GetPredictionResults(predictions)
    assert className == "NotRelevant"
    assert res == 0.6
    print(f"Target class '{className}' is {res}")
    predictions = np.array([0.2, 0.2, 0.2])
    className, res = GetPredictionResults(predictions)
    assert className == None
    assert res == None
    print(f"Target class '{className}' is {res}")
